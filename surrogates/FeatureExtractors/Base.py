import torch
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Any, Callable, Dict, Optional
from kmeans_pytorch import kmeans
import torch.nn.functional as F
from torchvision import transforms as T
import sys
import os
from contextlib import contextmanager
import numpy as np

def _session_caches() -> list:
    
    foa_root     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    foa_parent   = os.path.dirname(foa_root)     
    session_root = os.path.dirname(foa_parent)  
    return [
        os.path.join(session_root, "physpatch", ".hf_cache"),
        os.path.join(session_root, "20260424_104919_19_5", "grad_cam_comparison", ".hf_cache"),
        os.path.join(foa_root, ".hf_cache"),
    ]

def get_hf_cache_dir() -> str:
    for env_key in ("HUGGINGFACE_HUB_CACHE", "HF_HOME", "TRANSFORMERS_CACHE"):
        val = os.environ.get(env_key, "").strip()
        if val:
            os.makedirs(val, exist_ok=True)
            return val
    for path in _session_caches():
        if os.path.isdir(path) and os.access(path, os.R_OK | os.W_OK):
            return path
    foa_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default = os.path.join(foa_root, ".hf_cache")
    os.makedirs(default, exist_ok=True)
    return default

def find_hf_cache_for_model(model_id: str) -> str:
    for env_key in ("HUGGINGFACE_HUB_CACHE", "HF_HOME", "TRANSFORMERS_CACHE"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return val
    dir_name = "models--" + model_id.replace("/", "--")
    for cache in _session_caches():
        if os.path.isdir(os.path.join(cache, dir_name)):
            return cache
        if os.path.isdir(os.path.join(cache, "hub", dir_name)):
            return os.path.join(cache, "hub")
    return get_hf_cache_dir()

@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def patch_topk_informative(
    embedding_img: Tensor,
    k: int,
    mode: str = "center_dist",
    token_priority: Optional[Tensor] = None,
    priority_blend: float = 1.0,
) -> Tensor:
    if embedding_img.dim() == 1:
        embedding_img = embedding_img.unsqueeze(0)
    n = int(embedding_img.shape[0])
    k = min(int(k), max(1, n))
    if mode == "norm":
        emb_score = embedding_img.norm(dim=-1)
    elif mode == "center_dist":
        ctr = embedding_img.mean(dim=0, keepdim=True)
        emb_score = (embedding_img - ctr).norm(dim=-1)
    elif mode == "max_abs":
        emb_score = embedding_img.abs().max(dim=-1).values
    else:
        raise ValueError(f"Unknown patch_topk_informative mode '{mode}'")

    score = emb_score
    if token_priority is not None and token_priority.numel() == n:
        tp = token_priority.to(device=embedding_img.device, dtype=embedding_img.dtype).view(n)
        tp = tp - tp.min()
        tp = tp / (tp.max() + 1e-8)
        emb_n = emb_score - emb_score.min()
        emb_n = emb_n / (emb_n.max() + 1e-8)
        b = float(max(0.0, min(1.0, priority_blend)))
        score = b * tp + (1.0 - b) * emb_n

    _, idx = torch.topk(score, k)
    return embedding_img[idx].clone()

class BaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

class EnsembleFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor) -> Tensor:
        features = {}  
        for i, model in enumerate(self.extractors):
            features[i] = model(x).squeeze()
        return features

class EnsembleFeatureExtractor_ot(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor],cluster_number=5):
        super(EnsembleFeatureExtractor_ot, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.cluster_number = cluster_number
        self._active: Any = None

    def set_active_models(self, indices):
        self._active = indices

    def forward(self, x: Tensor) -> Tensor:
        features = {}
        features_local = {}
        active = self._active if self._active is not None else range(len(self.extractors))
        for i in active:
            model = self.extractors[i]
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            features[i] = x_tensor.squeeze()
            cluster_center = self.get_cluster_center(x_embedding[0],x.device).unsqueeze(0)
            features_local[i]=cluster_center

        return features,features_local

    def get_cluster_center(self, embedding_img,device):
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=self.cluster_number,
                distance='euclidean',
                device=device, 
            )
            cluster_center = cluster_center.to(device)
        return cluster_center

class EnsembleFeatureExtractor_ot_topk(EnsembleFeatureExtractor_ot):

    def __init__(
        self,
        extractors: List[BaseFeatureExtractor],
        cluster_number: int = 5,
        informative_mode: str = "center_dist",
        ubmb_priority_blend: float = 0.65,
    ):
        super().__init__(extractors, cluster_number=cluster_number)
        self.informative_mode = informative_mode
        self.ubmb_priority_blend = float(ubmb_priority_blend)
        self._ubmb_token_priority: Optional[Dict[int, Tensor]] = None

    def set_ubmb_token_priority(self, priorities: Optional[Dict[int, Tensor]]) -> None:
        self._ubmb_token_priority = priorities

    def forward(self, x: Tensor) -> Tensor:
        features: Dict[int, Tensor] = {}
        features_local: Dict[int, Tensor] = {}
        active = self._active if self._active is not None else range(len(self.extractors))
        for i in active:
            model = self.extractors[i]
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            features[i] = x_tensor.squeeze()
            prio: Optional[Tensor] = None
            if self._ubmb_token_priority is not None and i in self._ubmb_token_priority:
                cand = self._ubmb_token_priority[i]
                if cand is not None and int(cand.numel()) == int(x_embedding[0].shape[0]):
                    prio = cand.to(device=x_embedding.device, dtype=x_embedding.dtype)
            cluster_center = patch_topk_informative(
                x_embedding[0],
                self.cluster_number,
                self.informative_mode,
                token_priority=prio,
                priority_blend=self.ubmb_priority_blend,
            ).to(x.device)
            features_local[i] = cluster_center.unsqueeze(0)
        return features, features_local

    def get_cluster_center(self, embedding_img, device, token_priority: Optional[Tensor] = None):
        return patch_topk_informative(
            embedding_img,
            self.cluster_number,
            self.informative_mode,
            token_priority=token_priority,
            priority_blend=self.ubmb_priority_blend,
        ).to(device)

    def set_topk_patches(self, k: int) -> None:
        self.cluster_number = max(1, int(k))

class EnsembleFeatureExtractor_ot3(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor_ot3, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor) -> Tensor:
        features = {} 
        features_local = {}
        for i, model in enumerate(self.extractors):
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            features[i] = x_tensor.squeeze()
            cluster_center = self.get_cluster_center(x_embedding[0],x.device).unsqueeze(0)
            features_local[i]=cluster_center

        return features,features_local

    def get_cluster_center(self, embedding_img,device):
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=3,
                distance='euclidean',
                device=device, 
            )
            cluster_center = cluster_center.to(device)
        return cluster_center

class EnsembleFeatureLoss(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        for model in self.extractors:
            self.ground_truth.append(model(x).to(x.device))

    def __call__(self, feature_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss = 0
        for index, model in enumerate(self.extractors):
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            loss += torch.mean(torch.sum(feature * gt, dim=1))
            
        loss = loss / len(self.extractors)

        return loss

class EnsembleFeatureLoss_OT(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss_OT, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss = 0
        loss_local=0
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            feature_local = feature_local_dict[index].squeeze(0)
            loss_local += self.OT(gt_local, feature_local) * 2

            loss += torch.mean(torch.sum(feature * gt, dim=1))

        loss = loss / len(self.extractors)
        loss_local=loss_local / len(self.extractors)

        loss=loss+loss_local*0.1
        return loss

    def get_cluster_center(self, embedding_img):
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=5,
                distance='euclidean',
                device=torch.device('cuda'),
               
            )
            cluster_center = cluster_center.to('cuda')
        return cluster_center

    def OT(self, src_dis, tgt_dis):
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(30):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T

class EnsembleFeatureLoss_OT_Auto(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss_OT_Auto, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            feature_local = feature_local_dict[index].squeeze(0)
            local_loss = self.OT(gt_local, feature_local) * 2
            feat_loss = torch.mean(torch.sum(feature * gt, dim=1))

            loss_list.append(feat_loss)
            loss_local_list.append(local_loss)

        total_losses = [
            loss_list[i] + 0.1 * loss_local_list[i]
            for i in range(len(self.extractors))
        ]
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)
            
        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K  

        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

    def get_cluster_center(self, embedding_img):
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=5,
                distance='euclidean',
                device=torch.device('cuda'),
                 
            )
            cluster_center = cluster_center.to('cuda')
        return cluster_center

    def OT(self, src_dis, tgt_dis):
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(30):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T
    
class EnsembleFeatureLoss_OT_foa_attack(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor],cluster_number=5):
        super(EnsembleFeatureLoss_OT_foa_attack, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number
        self._alignment_prior = None
        self._active = None

    def set_active_models(self, indices):
        self._active = indices
        self.previous_loss_list = []

    def set_alignment_prior(self, prior):
        if prior is None:
            self._alignment_prior = None
            return
        p = np.asarray(prior, dtype=np.float64).reshape(-1)
        if p.shape[0] != len(self.extractors):
            raise ValueError(
                f"alignment prior length {p.shape[0]} != n extractors {len(self.extractors)}"
            )
        p = np.maximum(p, 1e-8)
        p = p / (p.mean() + 1e-12)
        self._alignment_prior = p

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding, x.device).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor],
                 feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        active = self._active if self._active is not None else list(range(len(self.extractors)))
        K = len(active)
        loss_list = []
        loss_local_list = []
        for index in active:
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt = self.ground_truth[index]
            feature = feature_dict[index].unsqueeze(0)
            feature_local = feature_local_dict[index].squeeze(0)
            local_loss = self.OT(gt_local, feature_local)
            feat_loss = self.OT(gt, feature)
            loss_list.append(feat_loss)
            loss_local_list.append(local_loss)

        total_losses = [
            loss_list[i] + 0.2 * loss_local_list[i]
            for i in range(K)
        ]

        if len(self.previous_loss_list) != K:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = [total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
                   for i in range(K)]

        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / 1.0)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K

        if self._alignment_prior is not None:
            prior = self._alignment_prior[np.array(active)] if self._active is not None \
                else self._alignment_prior
            weights_softmax = weights_softmax * prior
            s = weights_softmax.sum()
            if s > 1e-12:
                weights_softmax /= s
            weights_softmax *= K

        for i in range(K):
            self.previous_loss_list[i] = total_losses[i].detach()

        total_loss = sum(weights_softmax[i] * total_losses[i] for i in range(K))
        return total_loss

    def get_cluster_center(self, embedding_img,device):
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=self.cluster_number,
                distance='euclidean',
                device=device,
                 
            )
            cluster_center = cluster_center.to(device)
        return cluster_center

    def OT(self, src_dis, tgt_dis):
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(30):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T

class EnsembleFeatureLoss_OT_topk_patches(EnsembleFeatureLoss_OT_foa_attack):

    def __init__(
        self,
        extractors: List[BaseFeatureExtractor],
        cluster_number: int = 5,
        informative_mode: str = "center_dist",
        ubmb_priority_blend: float = 0.65,
    ):
        super().__init__(extractors, cluster_number=cluster_number)
        self.informative_mode = informative_mode
        self.ubmb_priority_blend = float(ubmb_priority_blend)
        self._ubmb_token_priority: Optional[Dict[int, Tensor]] = None

    def set_ubmb_token_priority(self, priorities: Optional[Dict[int, Tensor]]) -> None:
        self._ubmb_token_priority = priorities

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for idx, model in enumerate(self.extractors):
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            prio: Optional[Tensor] = None
            if self._ubmb_token_priority is not None and idx in self._ubmb_token_priority:
                cand = self._ubmb_token_priority[idx]
                if cand is not None and int(cand.numel()) == int(x_embedding.shape[0]):
                    prio = cand.to(device=x_embedding.device, dtype=x_embedding.dtype)
            cluster_center = patch_topk_informative(
                x_embedding,
                self.cluster_number,
                self.informative_mode,
                token_priority=prio,
                priority_blend=self.ubmb_priority_blend,
            ).to(x.device)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center.unsqueeze(0))

    def get_cluster_center(self, embedding_img, device, token_priority: Optional[Tensor] = None):
        return patch_topk_informative(
            embedding_img,
            self.cluster_number,
            self.informative_mode,
            token_priority=token_priority,
            priority_blend=self.ubmb_priority_blend,
        ).to(device)

    def set_topk_patches(self, k: int) -> None:
        self.cluster_number = max(1, int(k))

class EnsembleFeatureLoss_OT_ablation_wo_global(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number=5):
        super(EnsembleFeatureLoss_OT_ablation_wo_global, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding,x.device).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            feature_local = feature_local_dict[index].squeeze(0)
            local_loss = self.OT(gt_local, feature_local)
            feat_loss = torch.mean(torch.sum(feature * gt, dim=1))

            loss_list.append(feat_loss)
            loss_local_list.append(local_loss)

        total_losses = [
            loss_list[i] + 0.2 * loss_local_list[i]
            for i in range(len(self.extractors))
        ]
        
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)
            
        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K  

        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

    def get_cluster_center(self, embedding_img,device):
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=self.cluster_number,
                distance='euclidean',
                device=device,
                 
            )
            cluster_center = cluster_center.to(device)
        return cluster_center

    def OT(self, src_dis, tgt_dis):
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(30):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T
    
class EnsembleFeatureLoss_OT_ablation_wo_local(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number=5):
        super(EnsembleFeatureLoss_OT_ablation_wo_local, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding,x.device).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):
            
            gt = self.ground_truth[index]
            feature = feature_dict[index].unsqueeze(0)
            feat_loss = self.OT(gt,feature)

            loss_list.append(feat_loss)

        total_losses = [
            loss_list[i]
            for i in range(len(self.extractors))
        ]
       
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)
           
        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K  

        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

    def get_cluster_center(self, embedding_img,device):
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=self.cluster_number,
                distance='euclidean',
                device=device,
                  
            )
            cluster_center = cluster_center.to(device)
        return cluster_center

    def OT(self, src_dis, tgt_dis):
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(30):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T
    
class EnsembleFeatureLoss_OT_ablation_wo_dynamic(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number=5):
        super(EnsembleFeatureLoss_OT_ablation_wo_dynamic, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding,x.device).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):
            
            gt = self.ground_truth[index]
            feature = feature_dict[index].unsqueeze(0)
            feat_loss = self.OT(gt,feature)

            loss_list.append(feat_loss)

        total_losses = [
            loss_list[i]
            for i in range(len(self.extractors))
        ]

        total_loss = sum(total_losses)/len(total_losses)
        return total_loss

    def get_cluster_center(self, embedding_img,device):
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=self.cluster_number,
                distance='euclidean',
                device=device,
               
            )
            cluster_center = cluster_center.to(device)
        return cluster_center

    def OT(self, src_dis, tgt_dis):
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(30):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T