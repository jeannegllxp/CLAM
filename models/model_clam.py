import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
#from utils.utils import initialize_weights
import numpy as np

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net(nn.Module):

	def __init__(self, L=768, D=256, dropout=False, n_classes=1):
		super(Attn_Net, self).__init__()
		self.module = [
			nn.Linear(L, D),
			nn.Tanh()]

		if dropout:
			self.module.append(nn.Dropout(0.25))

		self.module.append(nn.Linear(D, n_classes))

		self.module = nn.Sequential(*self.module)

	def forward(self, x):
		return self.module(x), x  # N x n_classes


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net_Gated(nn.Module):
	def __init__(self, L=768, D=256, dropout=False, n_classes=1):
		super(Attn_Net_Gated, self).__init__()
		self.attention_a = [
			nn.Linear(L, D),
			nn.Tanh()]

		self.attention_b = [nn.Linear(L, D),
		                    nn.Sigmoid()]
		if dropout:
			self.attention_a.append(nn.Dropout(0.25))
			self.attention_b.append(nn.Dropout(0.25))

		self.attention_a = nn.Sequential(*self.attention_a)
		self.attention_b = nn.Sequential(*self.attention_b)

		self.attention_c = nn.Linear(D, n_classes)

	def forward(self, x):
		a = self.attention_a(x)
		b = self.attention_b(x)
		A = a.mul(b)
		A = self.attention_c(A)  # N x n_classes
		return A, x


"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""


class CLAM_SB(nn.Module):
	def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
	             instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=768):
		super(CLAM_SB, self).__init__()
		self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}  # Input L is size[0]
		size = self.size_dict[size_arg]
		fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
		if dropout:
			fc.append(nn.Dropout(0.25))
		if gate:
			attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
		else:
			attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
		fc.append(attention_net)
		self.attention_net = nn.Sequential(*fc)
		self.classifiers = nn.Linear(size[1], n_classes)
		instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
		self.instance_classifiers = nn.ModuleList(instance_classifiers)
		self.k_sample = k_sample
		self.instance_loss_fn = instance_loss_fn
		self.n_classes = n_classes
		self.subtyping = subtyping

		initialize_weights(self)

	def relocate(self):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.attention_net = self.attention_net.to(device)
		self.classifiers = self.classifiers.to(device)
		self.instance_classifiers = self.instance_classifiers.to(device)

	@staticmethod
	def create_positive_targets(length, device):
		return torch.full((length,), 1, device=device).long()

	@staticmethod
	def create_negative_targets(length, device):
		return torch.full((length,), 0, device=device).long()

	# instance-level evaluation for in-the-class attention branch
	def inst_eval(self, A, h, classifier):
		device = h.device
		if len(A.shape) == 1:
			A = A.view(1, -1)

		num_available_instances = A.size(1)
		current_k_sample = min(self.k_sample, num_available_instances)

		if current_k_sample == 0:
			empty_preds = torch.empty(0, dtype=torch.long, device=device)
			empty_targets = torch.empty(0, dtype=torch.long, device=device)
			return empty_loss, empty_preds, empty_targets

		top_p_ids = torch.topk(A, current_k_sample, dim=1)[1][-1]
		top_p = torch.index_select(h, dim=0, index=top_p_ids)
		top_n_ids = torch.topk(-A, current_k_sample, dim=1)[1][-1]
		top_n = torch.index_select(h, dim=0, index=top_n_ids)
		p_targets = self.create_positive_targets(current_k_sample, device)
		n_targets = self.create_negative_targets(current_k_sample, device)

		all_targets = torch.cat([p_targets, n_targets], dim=0)
		all_instances = torch.cat([top_p, top_n], dim=0)
		logits = classifier(all_instances)
		all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
		instance_loss = self.instance_loss_fn(logits, all_targets)
		return instance_loss, all_preds, all_targets

	# instance-level evaluation for out-of-the-class attention branch
	def inst_eval_out(self, A, h, classifier):
		device = h.device
		if len(A.shape) == 1:
			A = A.view(1, -1)

		num_available_instances = A.size(1)
		current_k_sample = min(self.k_sample, num_available_instances)

		if current_k_sample == 0:
			empty_loss = torch.tensor(0.0, device=device, requires_grad=True)
			empty_preds = torch.empty(0, dtype=torch.long, device=device)
			empty_targets = torch.empty(0, dtype=torch.long, device=device)
			return empty_loss, empty_preds, empty_targets

		top_p_ids = torch.topk(A, current_k_sample, dim=1)[1][-1]
		top_p = torch.index_select(h, dim=0, index=top_p_ids)
		p_targets = self.create_negative_targets(current_k_sample, device)
		logits = classifier(top_p)
		p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
		instance_loss = self.instance_loss_fn(logits, p_targets)
		return instance_loss, p_preds, p_targets

	def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
		device = h.device
		# A_before_transpose: N x 1 (K_attn_out is 1 for SB), h_transformed: N x L (L is size[1])
		A_before_transpose, h_transformed = self.attention_net(h)
		A_raw = torch.transpose(A_before_transpose, 1, 0)  # A_raw: 1 x N

		if attention_only:
			return A_raw  # Return raw attention scores before softmax

		A_softmax = F.softmax(A_raw, dim=1)  # softmax over N

		if instance_eval:  # instance_eval uses A_raw (pre-softmax)
			total_inst_loss = 0.0
			all_preds = []
			all_targets = []

			if label is not None and label.ndim == 0:  # Ensure label is 1D for one_hot
				label = label.unsqueeze(0)
			inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label

			if self.n_classes > 1 and inst_labels.ndim == 0:  # Handle case where squeeze might make it scalar
				inst_labels_one_hot = torch.zeros(self.n_classes, device=device, dtype=torch.long)
				inst_labels_one_hot[inst_labels.item()] = 1
				inst_labels = inst_labels_one_hot

			for i in range(len(self.instance_classifiers)):
				inst_label = inst_labels[i].item()
				classifier = self.instance_classifiers[i]
				# A_raw is (1, N) for SB. Pass this to inst_eval. h_transformed is (N, size[1])
				if inst_label == 1:  # in-the-class:
					instance_loss, preds, targets = self.inst_eval(A_raw, h_transformed,  # Use A_raw and h_transformed
					                                               classifier)
					all_preds.extend(preds.cpu().numpy())
					all_targets.extend(targets.cpu().numpy())
				else:  # out-of-the-class
					if self.subtyping:
						instance_loss, preds, targets = self.inst_eval_out(A_raw, h_transformed,
						                                                   classifier)  # Use A_raw
						all_preds.extend(preds.cpu().numpy())
						all_targets.extend(targets.cpu().numpy())
					else:
						continue
				total_inst_loss += instance_loss

			if self.subtyping and len(self.instance_classifiers) > 0:
				total_inst_loss /= len(self.instance_classifiers)
			elif not self.subtyping:
				# For non-subtyping, instance loss is only for the positive class, so no averaging over classifiers
				pass

		# M is calculated using A_softmax
		M = torch.mm(A_softmax, h_transformed)  # A_softmax is (1,N), h_transformed is (N, size[1]). M is (1, size[1])
		logits = self.classifiers(M)
		Y_hat = torch.topk(logits, 1, dim=1)[1]
		Y_prob = F.softmax(logits, dim=1)

		results_dict = {}
		if instance_eval:
			results_dict = {'instance_loss': total_inst_loss,
			                'inst_labels': np.array(all_targets) if len(all_targets) > 0 else np.array([]),
			                'inst_preds': np.array(all_preds) if len(all_preds) > 0 else np.array([])}

		if return_features:
			results_dict.update({'features': M})

		# A_raw is returned for consistency (e.g. for attention heatmaps)
		return logits, Y_prob, Y_hat, A_raw, results_dict


class CLAM_SB_PLIP_FEATURES(nn.Module):
	def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
	             instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
		super(CLAM_SB_PLIP_FEATURES, self).__init__()
		self.size_dict = {"small": [512, 512, 256], "big": [512, 512, 384]}
		size = self.size_dict[size_arg]
		fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
		if dropout:
			fc.append(nn.Dropout(0.25))
		if gate:
			attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
		else:
			attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
		fc.append(attention_net)
		self.attention_net = nn.Sequential(*fc)
		self.classifiers = nn.Linear(size[1], n_classes)
		instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
		self.instance_classifiers = nn.ModuleList(instance_classifiers)
		self.k_sample = k_sample
		self.instance_loss_fn = instance_loss_fn
		self.n_classes = n_classes
		self.subtyping = subtyping

		initialize_weights(self)

	def relocate(self):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.attention_net = self.attention_net.to(device)
		self.classifiers = self.classifiers.to(device)
		self.instance_classifiers = self.instance_classifiers.to(device)

	@staticmethod
	def create_positive_targets(length, device):
		return torch.full((length,), 1, device=device).long()

	@staticmethod
	def create_negative_targets(length, device):
		return torch.full((length,), 0, device=device).long()

	# instance-level evaluation for in-the-class attention branch
	def inst_eval(self, A, h, classifier):  # A is A_raw
		device = h.device
		if len(A.shape) == 1:
			A = A.view(1, -1)

		num_available_instances = A.size(1)
		current_k_sample = min(self.k_sample, num_available_instances)

		if current_k_sample == 0:
			empty_loss = torch.tensor(0.0, device=device, requires_grad=True)
			empty_preds = torch.empty(0, dtype=torch.long, device=device)
			empty_targets = torch.empty(0, dtype=torch.long, device=device)
			return empty_loss, empty_preds, empty_targets

		top_p_ids = torch.topk(A, current_k_sample, dim=1)[1][-1]
		top_p = torch.index_select(h, dim=0, index=top_p_ids)
		top_n_ids = torch.topk(-A, current_k_sample, dim=1)[1][-1]
		top_n = torch.index_select(h, dim=0, index=top_n_ids)
		p_targets = self.create_positive_targets(current_k_sample, device)
		n_targets = self.create_negative_targets(current_k_sample, device)

		all_targets = torch.cat([p_targets, n_targets], dim=0)
		all_instances = torch.cat([top_p, top_n], dim=0)
		logits = classifier(all_instances)
		all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
		instance_loss = self.instance_loss_fn(logits, all_targets)
		return instance_loss, all_preds, all_targets

	# instance-level evaluation for out-of-the-class attention branch
	def inst_eval_out(self, A, h, classifier):  # A is A_raw
		device = h.device
		if len(A.shape) == 1:
			A = A.view(1, -1)

		num_available_instances = A.size(1)
		current_k_sample = min(self.k_sample, num_available_instances)

		if current_k_sample == 0:
			empty_loss = torch.tensor(0.0, device=device, requires_grad=True)
			empty_preds = torch.empty(0, dtype=torch.long, device=device)
			empty_targets = torch.empty(0, dtype=torch.long, device=device)
			return empty_loss, empty_preds, empty_targets

		top_p_ids = torch.topk(A, current_k_sample, dim=1)[1][-1]
		top_p = torch.index_select(h, dim=0, index=top_p_ids)
		p_targets = self.create_negative_targets(current_k_sample, device)
		logits = classifier(top_p)
		p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
		instance_loss = self.instance_loss_fn(logits, p_targets)
		return instance_loss, p_preds, p_targets

	def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
		device = h.device
		A_before_transpose, h_transformed = self.attention_net(h)  # A: NxK (K=1 for SB), h is NxL
		A_raw = torch.transpose(A_before_transpose, 1, 0)  # KxN (1xN for SB)

		if attention_only:
			return A_raw  # Return raw attention scores

		A_softmax = F.softmax(A_raw, dim=1)  # softmax over N

		if instance_eval:
			total_inst_loss = 0.0
			all_preds = []
			all_targets = []

			if label is not None and label.ndim == 0:
				label = label.unsqueeze(0)
			inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()

			if self.n_classes > 1 and inst_labels.ndim == 0:
				inst_labels_one_hot = torch.zeros(self.n_classes, device=device, dtype=torch.long)
				inst_labels_one_hot[inst_labels.item()] = 1
				inst_labels = inst_labels_one_hot

			for i in range(len(self.instance_classifiers)):
				inst_label = inst_labels[i].item()
				classifier = self.instance_classifiers[i]
				if inst_label == 1:
					instance_loss, preds, targets = self.inst_eval(A_raw, h_transformed, classifier)  # Use A_raw
					all_preds.extend(preds.cpu().numpy())
					all_targets.extend(targets.cpu().numpy())
				else:
					if self.subtyping:
						instance_loss, preds, targets = self.inst_eval_out(A_raw, h_transformed,
						                                                   classifier)  # Use A_raw
						all_preds.extend(preds.cpu().numpy())
						all_targets.extend(targets.cpu().numpy())
					else:
						continue
				total_inst_loss += instance_loss

			if self.subtyping and len(self.instance_classifiers) > 0:
				total_inst_loss /= len(self.instance_classifiers)
			elif not self.subtyping:
				pass

		M = torch.mm(A_softmax, h_transformed)
		logits = self.classifiers(M)
		Y_hat = torch.topk(logits, 1, dim=1)[1]
		Y_prob = F.softmax(logits, dim=1)

		results_dict = {}
		if instance_eval:
			results_dict = {'instance_loss': total_inst_loss,
			                'inst_labels': np.array(all_targets) if len(all_targets) > 0 else np.array([]),
			                'inst_preds': np.array(all_preds) if len(all_preds) > 0 else np.array([])}

		if return_features:
			results_dict.update({'features': M})
		return logits, Y_prob, Y_hat, A_raw, results_dict


class CLAM_MB(CLAM_SB):
	def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
	             instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
		nn.Module.__init__(self)
		self.size_dict = {"small": [512, 512, 256], "big": [512, 512, 384]}
		size = self.size_dict[size_arg]
		fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
		if dropout:
			fc.append(nn.Dropout(0.25))
		if gate:
			attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout,
			                               n_classes=n_classes)
		else:
			attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout,
			                         n_classes=n_classes)
		fc.append(attention_net)
		self.attention_net = nn.Sequential(*fc)
		bag_classifiers = [nn.Linear(size[1], 1) for i in
		                   range(n_classes)]
		self.classifiers = nn.ModuleList(bag_classifiers)
		instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
		self.instance_classifiers = nn.ModuleList(instance_classifiers)
		self.k_sample = k_sample
		self.instance_loss_fn = instance_loss_fn
		self.n_classes = n_classes
		self.subtyping = subtyping
		initialize_weights(self)

	def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
		device = h.device
		A_before_transpose, h_transformed = self.attention_net(
			h)  # A_before_transpose: N x n_classes, h_transformed: N x size[1]
		A_raw = torch.transpose(A_before_transpose, 1, 0)  # A_raw: n_classes x N

		if attention_only:
			return A_raw  # Return raw attention scores

		A_softmax = F.softmax(A_raw, dim=1)  # softmax over N for each attention branch

		if instance_eval:  # instance_eval uses A_raw (pre-softmax)
			total_inst_loss = 0.0
			all_preds = []
			all_targets = []

			if label is not None and label.ndim == 0:
				label = label.unsqueeze(0)
			inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()

			if self.n_classes > 1 and inst_labels.ndim == 0:
				inst_labels_one_hot = torch.zeros(self.n_classes, device=device, dtype=torch.long)
				inst_labels_one_hot[inst_labels.item()] = 1
				inst_labels = inst_labels_one_hot

			for i in range(len(self.instance_classifiers)):
				inst_label = inst_labels[i].item()
				classifier = self.instance_classifiers[i]
				# A_raw[i] is (N), needs unsqueeze to be (1,N) for inst_eval
				current_A_raw_branch = A_raw[i].unsqueeze(0)
				if inst_label == 1:
					instance_loss, preds, targets = self.inst_eval(current_A_raw_branch, h_transformed, classifier)
					all_preds.extend(preds.cpu().numpy())
					all_targets.extend(targets.cpu().numpy())
				else:
					if self.subtyping:
						instance_loss, preds, targets = self.inst_eval_out(current_A_raw_branch, h_transformed,
						                                                   classifier)
						all_preds.extend(preds.cpu().numpy())
						all_targets.extend(targets.cpu().numpy())
					else:
						continue
				total_inst_loss += instance_loss

			if self.subtyping and len(self.instance_classifiers) > 0:
				total_inst_loss /= len(self.instance_classifiers)

		# M is calculated using A_softmax
		M = torch.mm(A_softmax, h_transformed)  # (n_classes x N) x (N x size[1]) -> (n_classes x size[1])

		logits = torch.empty(1, self.n_classes).float().to(device)
		for c in range(self.n_classes):
			logits[0, c] = self.classifiers[c](M[c].unsqueeze(0))

		Y_hat = torch.topk(logits, 1, dim=1)[1]
		Y_prob = F.softmax(logits, dim=1)

		results_dict = {}
		if instance_eval:
			results_dict = {'instance_loss': total_inst_loss,
			                'inst_labels': np.array(all_targets) if len(all_targets) > 0 else np.array([]),
			                'inst_preds': np.array(all_preds) if len(all_preds) > 0 else np.array([])}

		if return_features:
			results_dict.update({'features': M})
		return logits, Y_prob, Y_hat, A_raw, results_dict


class DIP_Module(nn.Module):
	"""
    Dynamic Instance Pruning (DIP) Module.
    Selects top-K instances based on learned scores.
    """

	def __init__(self, L=768, hidden_dim=128, K=64):
		super(DIP_Module, self).__init__()
		self.K = K
		self.scorer = nn.Sequential(
			nn.Linear(L, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)
		initialize_weights(self)

	def forward(self, h):
		# h: (N x L), N is number of instances in the bag, L is feature_dim
		if h.shape[0] == 0:  # No instances in the bag
			return h

		scores = self.scorer(h)  # (N x 1)

		# Determine actual K to use (cannot be more than N)
		current_k = min(self.K, h.shape[0])

		if current_k == 0:
			return torch.empty(0, h.shape[1], device=h.device, dtype=h.dtype)

		# Select top-K instances based on scores
		# scores.squeeze(1) gives (N), top_indices gives (current_k)
		_, top_indices = torch.topk(scores.squeeze(1), k=current_k, dim=0)

		h_selected = torch.index_select(h, dim=0, index=top_indices)  # (current_k x L)
		return h_selected


class CLAM_DIP_SB(nn.Module):
	"""
    CLAM with Dynamic Instance Pruning (Single Branch).
    Applies DIP before the standard CLAM attention mechanism.
    """

	def __init__(self, original_feature_dim=768, dip_k=64, dip_hidden_dim=128,  # DIP specific params
	             gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,  # CLAM specific
	             instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
		super(CLAM_DIP_SB, self).__init__()

		self.original_feature_dim = original_feature_dim
		self.dip_k = dip_k

		# DIP Module
		self.dip_module = DIP_Module(L=original_feature_dim, hidden_dim=dip_hidden_dim, K=dip_k)

		# CLAM SB specific parts, adapted to use original_feature_dim as input to the first Linear layer
		# The first element in size_dict lists is the input feature dimension to the CLAM part
		self.size_dict_clam = {"small": [original_feature_dim, 512, 256],
		                       "big": [original_feature_dim, 512, 384]}
		size = self.size_dict_clam[size_arg]

		# CLAM's main feature transformation and attention pathway
		fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
		if dropout:
			fc.append(nn.Dropout(0.25))
		if gate:
			attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout,
			                               n_classes=1)
		else:
			attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
		fc.append(attention_net)
		self.attention_net = nn.Sequential(*fc)

		# Classifiers
		self.classifiers = nn.Linear(size[1], n_classes)
		instance_classifiers = [nn.Linear(size[1], 2) for _ in
		                        range(n_classes)]
		self.instance_classifiers = nn.ModuleList(instance_classifiers)

		self.k_sample = k_sample
		self.instance_loss_fn = instance_loss_fn
		self.n_classes = n_classes
		self.subtyping = subtyping

		initialize_weights(self)

	def relocate(self):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.dip_module = self.dip_module.to(device)
		self.attention_net = self.attention_net.to(device)
		self.classifiers = self.classifiers.to(device)
		self.instance_classifiers = self.instance_classifiers.to(device)

	@staticmethod
	def create_positive_targets(length, device):
		return torch.full((length,), 1, device=device).long()

	@staticmethod
	def create_negative_targets(length, device):
		return torch.full((length,), 0, device=device).long()

	def inst_eval(self, A_raw_selected, h_transformed_selected,
	              classifier):  # A_raw_selected is (1, K_sel), h_transformed_selected is (K_sel, size[1])
		device = h_transformed_selected.device
		if len(A_raw_selected.shape) == 1:
			A_raw_selected = A_raw_selected.view(1, -1)

		num_available_instances = A_raw_selected.size(1)
		current_k_sample = min(self.k_sample, num_available_instances)

		if current_k_sample == 0:
			empty_loss = torch.tensor(0.0, device=device, requires_grad=True)
			empty_preds = torch.empty(0, dtype=torch.long, device=device)
			empty_targets = torch.empty(0, dtype=torch.long, device=device)
			return empty_loss, empty_preds, empty_targets

		top_p_ids = torch.topk(A_raw_selected, current_k_sample, dim=1)[1][-1]
		top_p = torch.index_select(h_transformed_selected, dim=0, index=top_p_ids)

		top_n_ids = torch.topk(-A_raw_selected, current_k_sample, dim=1)[1][-1]
		top_n = torch.index_select(h_transformed_selected, dim=0, index=top_n_ids)

		p_targets = self.create_positive_targets(current_k_sample, device)
		n_targets = self.create_negative_targets(current_k_sample, device)

		all_targets = torch.cat([p_targets, n_targets], dim=0)
		all_instances = torch.cat([top_p, top_n], dim=0)
		logits = classifier(all_instances)
		all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
		instance_loss = self.instance_loss_fn(logits, all_targets)
		return instance_loss, all_preds, all_targets

	def inst_eval_out(self, A_raw_selected, h_transformed_selected, classifier):
		device = h_transformed_selected.device
		if len(A_raw_selected.shape) == 1:
			A_raw_selected = A_raw_selected.view(1, -1)

		num_available_instances = A_raw_selected.size(1)
		current_k_sample = min(self.k_sample, num_available_instances)

		if current_k_sample == 0:
			empty_loss = torch.tensor(0.0, device=device, requires_grad=True)
			empty_preds = torch.empty(0, dtype=torch.long, device=device)
			empty_targets = torch.empty(0, dtype=torch.long, device=device)
			return empty_loss, empty_preds, empty_targets

		top_p_ids = torch.topk(A_raw_selected, current_k_sample, dim=1)[1][-1]
		top_p = torch.index_select(h_transformed_selected, dim=0, index=top_p_ids)

		p_targets = self.create_negative_targets(current_k_sample, device)
		logits = classifier(top_p)
		p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
		instance_loss = self.instance_loss_fn(logits, p_targets)
		return instance_loss, p_preds, p_targets

	def forward(self, h_original, label=None, instance_eval=False, return_features=False, attention_only=False):
		# h_original: (N_original x original_feature_dim)
		device = h_original.device

		# 1. Apply DIP Module to select K instances
		# h_selected: (K_selected x original_feature_dim)
		h_selected = self.dip_module(h_original)

		if h_selected.shape[0] == 0:
			size_1_dim = self.size_dict_clam[list(self.size_dict_clam.keys())[0]][1]  # Get typical feature dim
			logits = torch.zeros(1, self.n_classes).to(device)
			Y_prob = F.softmax(logits, dim=1)
			Y_hat = torch.topk(logits, 1, dim=1)[1]
			A_raw_output = torch.empty(1, 0, device=device)  # Output A_raw should be (1, K_selected)
			results_dict = {}
			if instance_eval:
				results_dict = {'instance_loss': torch.tensor(0.0, device=device, requires_grad=True),
				                'inst_labels': np.array([]),
				                'inst_preds': np.array([])}
			if return_features:
				results_dict.update({'features': torch.zeros(1, size_1_dim).to(device)})
			return logits, Y_prob, Y_hat, A_raw_output, results_dict

		# 2. Pass K_selected features through CLAM's attention network
		# A_scores_before_transpose: (K_selected x 1 for SB), h_transformed_selected: (K_selected x size[1])
		A_scores_before_transpose, h_transformed_selected = self.attention_net(h_selected)
		A_raw_selected = torch.transpose(A_scores_before_transpose, 1, 0)  # A_raw_selected: (1 x K_selected for SB)

		if attention_only:
			return A_raw_selected  # These are raw scores for K_selected items

		A_softmax_selected = F.softmax(A_raw_selected, dim=1)

		# Instance-level evaluation (operates on K_selected instances and their transformed features, using A_raw_selected)
		if instance_eval:
			total_inst_loss = 0.0
			all_preds = []
			all_targets = []

			if label is not None and label.ndim == 0:
				label = label.unsqueeze(0)
			inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()

			if self.n_classes > 1 and inst_labels.ndim == 0:
				inst_labels_one_hot = torch.zeros(self.n_classes, device=device, dtype=torch.long)
				inst_labels_one_hot[inst_labels.item()] = 1
				inst_labels = inst_labels_one_hot

			for i in range(len(self.instance_classifiers)):
				inst_label = inst_labels[i].item()
				classifier = self.instance_classifiers[i]
				# Pass A_raw_selected (1 x K_selected) and h_transformed_selected (K_selected x size[1])
				if inst_label == 1:
					instance_loss, preds, targets = self.inst_eval(A_raw_selected, h_transformed_selected, classifier)
					all_preds.extend(preds.cpu().numpy())
					all_targets.extend(targets.cpu().numpy())
				else:
					if self.subtyping:
						instance_loss, preds, targets = self.inst_eval_out(A_raw_selected, h_transformed_selected,
						                                                   classifier)
						all_preds.extend(preds.cpu().numpy())
						all_targets.extend(targets.cpu().numpy())
					else:
						continue
				total_inst_loss += instance_loss

			if self.subtyping and len(self.instance_classifiers) > 0:
				total_inst_loss /= len(self.instance_classifiers)
			elif not self.subtyping:
				pass

		# Aggregation: M is (1 x size[1]) for SB
		M = torch.mm(A_softmax_selected, h_transformed_selected)

		logits = self.classifiers(M)
		Y_hat = torch.topk(logits, 1, dim=1)[1]
		Y_prob = F.softmax(logits, dim=1)

		results_dict = {}
		if instance_eval:
			results_dict = {'instance_loss': total_inst_loss,
			                'inst_labels': np.array(all_targets) if len(all_targets) > 0 else np.array([]),
			                'inst_preds': np.array(all_preds) if len(all_preds) > 0 else np.array([])}

		if return_features:
			results_dict.update({'features': M})

		# A_raw_selected is (1, K_selected)
		return logits, Y_prob, Y_hat, A_raw_selected, results_dict


class CLAM_StochasticAttention_SB(nn.Module):
	def __init__(self, original_feature_dim=768, attention_sample_k=0,  # New params
	             gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,  # CLAM SB params
	             instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=768):
		super(CLAM_StochasticAttention_SB, self).__init__()
		self.original_feature_dim = original_feature_dim
		self.attention_sample_k = attention_sample_k

		# Size dictionary uses original_feature_dim as input to the first Linear layer
		self.size_dict = {"small": [original_feature_dim, 512, 256],
		                  "big": [original_feature_dim, 512, 384]}
		size = self.size_dict[size_arg]

		fc = [nn.Linear(size[0], size[1]), nn.ReLU()]  # size[0] is original_feature_dim
		if dropout:
			fc.append(nn.Dropout(0.25))
		if gate:
			attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
		else:
			attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
		fc.append(attention_net)
		self.attention_net = nn.Sequential(*fc)

		self.classifiers = nn.Linear(size[1], n_classes)
		instance_classifiers = [nn.Linear(size[1], 2) for _ in range(n_classes)]
		self.instance_classifiers = nn.ModuleList(instance_classifiers)

		self.k_sample = k_sample  # For deterministic instance evaluation
		self.instance_loss_fn = instance_loss_fn
		self.n_classes = n_classes
		self.subtyping = subtyping

		initialize_weights(self)

	def relocate(self):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.attention_net = self.attention_net.to(device)
		self.classifiers = self.classifiers.to(device)
		self.instance_classifiers = self.instance_classifiers.to(device)

	@staticmethod
	def create_positive_targets(length, device):
		return torch.full((length,), 1, device=device).long()

	@staticmethod
	def create_negative_targets(length, device):
		return torch.full((length,), 0, device=device).long()

	# Instance-level evaluation remains deterministic, based on CLAM_SB's method
	# It uses A_raw (raw attention scores before softmax and sampling) and h_transformed
	def inst_eval(self, A_raw_for_inst_eval, h_transformed, classifier):
		device = h_transformed.device
		if len(A_raw_for_inst_eval.shape) == 1:
			A_raw_for_inst_eval = A_raw_for_inst_eval.view(1, -1)

		num_available_instances = A_raw_for_inst_eval.size(1)
		current_k_sample = min(self.k_sample, num_available_instances)

		if current_k_sample == 0:
			empty_loss = torch.tensor(0.0, device=device, requires_grad=True)
			empty_preds = torch.empty(0, dtype=torch.long, device=device)
			empty_targets = torch.empty(0, dtype=torch.long, device=device)
			return empty_loss, empty_preds, empty_targets

		top_p_ids = torch.topk(A_raw_for_inst_eval, current_k_sample, dim=1)[1][-1]
		top_p = torch.index_select(h_transformed, dim=0, index=top_p_ids)
		top_n_ids = torch.topk(-A_raw_for_inst_eval, current_k_sample, dim=1)[1][-1]
		top_n = torch.index_select(h_transformed, dim=0, index=top_n_ids)

		p_targets = self.create_positive_targets(current_k_sample, device)
		n_targets = self.create_negative_targets(current_k_sample, device)

		all_targets = torch.cat([p_targets, n_targets], dim=0)
		all_instances = torch.cat([top_p, top_n], dim=0)
		logits = classifier(all_instances)
		all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
		instance_loss = self.instance_loss_fn(logits, all_targets)
		return instance_loss, all_preds, all_targets

	def inst_eval_out(self, A_raw_for_inst_eval, h_transformed, classifier):
		device = h_transformed.device
		if len(A_raw_for_inst_eval.shape) == 1:
			A_raw_for_inst_eval = A_raw_for_inst_eval.view(1, -1)

		num_available_instances = A_raw_for_inst_eval.size(1)
		current_k_sample = min(self.k_sample, num_available_instances)

		if current_k_sample == 0:
			empty_loss = torch.tensor(0.0, device=device, requires_grad=True)
			empty_preds = torch.empty(0, dtype=torch.long, device=device)
			empty_targets = torch.empty(0, dtype=torch.long, device=device)
			return empty_loss, empty_preds, empty_targets

		top_p_ids = torch.topk(A_raw_for_inst_eval, current_k_sample, dim=1)[1][-1]
		top_p = torch.index_select(h_transformed, dim=0, index=top_p_ids)
		p_targets = self.create_negative_targets(current_k_sample, device)
		logits = classifier(top_p)
		p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
		instance_loss = self.instance_loss_fn(logits, p_targets)
		return instance_loss, p_preds, p_targets

	def forward(self, h_input, label=None, instance_eval=False, return_features=False, attention_only=False):
		# h_input: (N_instances x original_feature_dim)
		device = h_input.device

		# A_from_net: (N_instances x 1 for SB), h_transformed: (N_instances x size[1])
		A_from_net, h_transformed = self.attention_net(h_input)
		# A_raw_output is (1 x N_instances), these are raw scores from attention network, before softmax.
		# This is used for deterministic inst_eval and is the consistent A_raw output for the model.
		A_raw_output = torch.transpose(A_from_net, 1, 0)

		if attention_only:
			return A_raw_output  # Return raw attention scores consistent with CLAM_SB output

		# A_prob is (1 x N_instances), probabilities for sampling or deterministic aggregation
		A_prob = F.softmax(A_raw_output, dim=1)

		# Stochastic sampling for bag aggregation M during training
		if self.training and self.attention_sample_k > 0 and h_transformed.shape[0] > 0:
			perform_stochastic_aggregation = True
			# Ensure enough instances for sampling without replacement
			if h_transformed.shape[0] < self.attention_sample_k:
				# Not enough instances to sample attention_sample_k, use all available deterministically for this batch
				# Or could sample with replacement if that was desired, but prompt implies no replacement by typical multinomial usage
				# print(f"Warning: Not enough instances ({h_transformed.shape[0]}) to sample {self.attention_sample_k}. Using deterministic aggregation for this batch.")
				perform_stochastic_aggregation = False  # Fallback to deterministic

			if perform_stochastic_aggregation:
				try:
					# A_prob.squeeze(0) is (N_instances), multinomial needs 1D prob input
					# Sample attention_sample_k indices based on A_prob
					sampled_indices = torch.multinomial(A_prob.squeeze(0), self.attention_sample_k, replacement=False)

					# Select corresponding transformed features and attention probabilities
					h_sampled = torch.index_select(h_transformed, dim=0,
					                               index=sampled_indices)  # (attention_sample_k x size[1])
					A_prob_sampled = torch.index_select(A_prob, dim=1,
					                                    index=sampled_indices)  # (1 x attention_sample_k)

					# Re-normalize selected A_prob_sampled to sum to 1 for weighted sum
					A_prob_sampled_renorm = A_prob_sampled / torch.sum(A_prob_sampled, dim=1, keepdim=True)

					M = torch.mm(A_prob_sampled_renorm,
					             h_sampled)  # (1 x attention_sample_k) x (attention_sample_k x size[1]) -> (1 x size[1])
				except RuntimeError as e:
					# Fallback to deterministic aggregation if multinomial fails (e.g., issues with probabilities, not enough unique items for no-replacement)
					# print(f"RuntimeError during multinomial sampling: {e}. Falling back to deterministic aggregation.")
					M = torch.mm(A_prob, h_transformed)  # Standard deterministic aggregation
			else:  # Not enough instances for the specified attention_sample_k
				M = torch.mm(A_prob, h_transformed)
		else:
			# Deterministic aggregation (evaluation mode or attention_sample_k <= 0 or no instances)
			if h_transformed.shape[
				0] == 0:  # Handle empty bag after potential upstream processing (e.g. DIP if this model was combined)
				# This model doesn't have DIP, so h_input -> h_transformed directly. If h_input is empty, h_transformed is empty.
				size_1_dim = self.size_dict[list(self.size_dict.keys())[0]][1]
				M = torch.zeros(1, size_1_dim).to(device)  # Default aggregated feature for empty bag
			else:
				M = torch.mm(A_prob, h_transformed)  # (1 x N) x (N x size[1]) -> (1 x size[1])

		logits = self.classifiers(M)
		Y_hat = torch.topk(logits, 1, dim=1)[1]
		Y_prob = F.softmax(logits, dim=1)

		results_dict = {}
		if instance_eval:  # instance_eval always uses A_raw_output and full h_transformed (deterministic)
			total_inst_loss = 0.0
			all_preds = []
			all_targets = []

			if label is not None and label.ndim == 0:
				label = label.unsqueeze(0)
			inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()

			if self.n_classes > 1 and inst_labels.ndim == 0:
				inst_labels_one_hot = torch.zeros(self.n_classes, device=device, dtype=torch.long)
				inst_labels_one_hot[inst_labels.item()] = 1
				inst_labels = inst_labels_one_hot

			# Check if h_transformed is empty before proceeding with inst_eval
			if h_transformed.shape[0] > 0:
				for i in range(len(self.instance_classifiers)):
					inst_label = inst_labels[i].item()
					classifier = self.instance_classifiers[i]
					if inst_label == 1:
						instance_loss, preds, targets = self.inst_eval(A_raw_output, h_transformed, classifier)
						all_preds.extend(preds.cpu().numpy())
						all_targets.extend(targets.cpu().numpy())
					else:
						if self.subtyping:
							instance_loss, preds, targets = self.inst_eval_out(A_raw_output, h_transformed, classifier)
							all_preds.extend(preds.cpu().numpy())
							all_targets.extend(targets.cpu().numpy())
						else:
							instance_loss = torch.tensor(0.0,
							                             device=device)  # No loss if not subtyping and not the positive class
					total_inst_loss += instance_loss

				if self.subtyping and len(self.instance_classifiers) > 0:
					total_inst_loss /= len(self.instance_classifiers)
			# Non-subtyping: instance loss is only for the positive class, no division by len(classifiers) needed here
			# as it's handled by the loop and only one branch contributes effectively.
			else:  # h_transformed is empty, no instances to evaluate
				total_inst_loss = torch.tensor(0.0, device=device)

			results_dict = {'instance_loss': total_inst_loss,
			                'inst_labels': np.array(all_targets) if len(all_targets) > 0 else np.array([]),
			                'inst_preds': np.array(all_preds) if len(all_preds) > 0 else np.array([])}

		if return_features:
			results_dict.update({'features': M})

		# Return A_raw_output (1xN, pre-softmax, pre-sampling) for consistency with other CLAM models
		return logits, Y_prob, Y_hat, A_raw_output, results_dict