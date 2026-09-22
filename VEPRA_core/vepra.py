import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from adv import AdversarialLoss
from prototype_confidence import PrototypeConfidence



class DynamicFreqAttention(nn.Module):

    def __init__(
        self,
        num_bands=5,
        input_dim=310,
        alpha_init=0.1,
        gate_hidden=64,
        channel_reduction=4,
    ):
        super().__init__()

        if gate_hidden <= 0 or channel_reduction <= 0:
            raise ValueError("DynamicFAP dimensions must be positive.")

        self.num_bands = num_bands
        self.num_channels = input_dim // num_bands

        
        self.global_band_weight = nn.Parameter(
            torch.ones(num_bands)
        )

        
        self.band_gate = nn.Sequential(
            nn.Linear(input_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, num_bands)
        )

        
        self.channel_attn = nn.Sequential(
            nn.Linear(
                self.num_channels,
                max(1, self.num_channels // channel_reduction),
            ),
            nn.ReLU(),
            nn.Linear(
                max(1, self.num_channels // channel_reduction),
                self.num_channels,
            ),
            nn.Sigmoid()
        )

        
        self.alpha = nn.Parameter(
            torch.tensor(float(alpha_init))
        )

    def forward(self, x):

        residual = x

        B = x.size(0)

        x_band = x.view(
            B,
            self.num_bands,
            self.num_channels
        )

        
        
        

        global_weight = F.softmax(
            self.global_band_weight,
            dim=0
        )

        
        
        

        dynamic_weight = F.softmax(
            self.band_gate(x),
            dim=1
        )

        band_weight = (
            global_weight.unsqueeze(0)
            * dynamic_weight
        )

        x_band = x_band * band_weight.unsqueeze(2)

        
        
        

        channel_feat = x_band.mean(dim=1)

        channel_weight = self.channel_attn(
            channel_feat
        )

        x_band = x_band * channel_weight.unsqueeze(1)

        
        
        

        x_band = x_band.reshape(B, -1)

        out = residual + self.alpha * x_band

        return out

    def get_parameters(self):
        return [
            {
                "params": self.parameters(),
                "lr_mult": 1
            }
        ]


class FreqAttention(nn.Module):
    
    def __init__(self, num_bands=5, input_dim=310, band_dim=64, dropout=0.1):
        super().__init__()
        self.num_bands = num_bands
        self.num_channels = input_dim // num_bands   

        
        self.band_proj = nn.Linear(self.num_channels, band_dim)        
        self.band_reproj = nn.Linear(band_dim, self.num_channels)      
        self.band_attn = nn.MultiheadAttention(embed_dim=band_dim, num_heads=4,
                                               dropout=dropout, batch_first=True)
        self.band_norm = nn.LayerNorm(band_dim)

        
        self.band_weight_raw = nn.Parameter(torch.zeros(num_bands))

        
        self.channel_attn = nn.Sequential(
            nn.Linear(self.num_channels * 2, self.num_channels // 4),
            nn.ReLU(),
            nn.Linear(self.num_channels // 4, self.num_channels),
            nn.Sigmoid()
        )

        
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B = x.size(0)
        
        x = x.view(B, self.num_bands, self.num_channels)

        
        band_feat = self.band_proj(x)                     
        band_feat_attn, _ = self.band_attn(band_feat, band_feat, band_feat)
        band_feat = self.band_norm(band_feat + band_feat_attn)  
        
        band_output = self.band_reproj(band_feat)         

        
        band_weights = F.softmax(self.band_weight_raw, dim=0)  
        band_output = band_output * band_weights.view(1, -1, 1)

        
        avg_pool = band_output.mean(dim=1)   
        max_pool, _ = band_output.max(dim=1) 
        channel_in = torch.cat([avg_pool, max_pool], dim=1)  
        channel_weights = self.channel_attn(channel_in)       
        band_output = band_output * channel_weights.unsqueeze(1)

        
        band_output = band_output.reshape(B, -1)   
        output = x.view(B, -1) + self.alpha * band_output

        return output

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]


class SupervisedContrastiveLoss(nn.Module): 
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feat: torch.Tensor, lbl: torch.Tensor) -> torch.Tensor:

        lbl = lbl.to(feat.device)
        
        feat = F.normalize(feat, p=2, dim=1)  

        
        sim = torch.mm(feat, feat.T)  

        
        mask = (lbl.unsqueeze(0) == lbl.unsqueeze(1)).float()
        mask.fill_diagonal_(0)  

        
        logits = sim / self.temperature

        log_prob = F.log_softmax(logits, dim=1)

        loss = -(log_prob * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return loss.mean()

class DomainAdaptationLoss(nn.Module):
    def __init__(self, loss_type="dann", **kwargs):
        super().__init__()
        if loss_type != "dann":
            raise ValueError("The public core release supports transfer_loss_type=dann.")
        self.loss_func = AdversarialLoss(**kwargs)

    def forward(self, source, target, **kwargs):
        return self.loss_func(source, target, **kwargs)

class SharedEncoder(nn.Module):

    def __init__(
            self,
            input_dim=310,
            hidden_1=64,
            hidden_2=64,
            use_freq_attn=False,
            use_dynamic_fap=False,
            fap_alpha_init=0.1,
            fap_gate_hidden=64,
            fap_channel_reduction=4):
        super().__init__()

        self.use_freq_attn = use_freq_attn
        self.use_dynamic_fap = use_dynamic_fap

        if use_freq_attn:
            if use_dynamic_fap:
                print(">>> Dynamic FAP Enabled")
                self.freq_attn = DynamicFreqAttention(
                    num_bands=5,
                    input_dim=input_dim,
                    alpha_init=fap_alpha_init,
                    gate_hidden=fap_gate_hidden,
                    channel_reduction=fap_channel_reduction,
                )
            else:
                self.freq_attn = FreqAttention(
                    num_bands=5,
                    input_dim=input_dim
                )

        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x):
        if self.use_freq_attn:
            x = self.freq_attn(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

    def forward_cssc_fc2_only(self, x):
        
        with torch.no_grad():
            if self.use_freq_attn:
                x = self.freq_attn(x)
            x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x.detach()))

    def get_parameters(self):
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]
        if self.use_freq_attn:
            params.extend(self.freq_attn.get_parameters())
        return params


class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, num_classes)

    def forward(self, feature):
        f1 = self.fc1(feature)
        f2 = self.fc2(f1)
        return self.fc3(f2)

    def predict(self, feature):
        with torch.no_grad():
            logits = F.softmax(self.forward(feature), dim=1)
            return torch.argmax(logits, axis=1)

    def get_parameters(self):
        return [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
            {"params": self.fc3.parameters(), "lr_mult": 1}
        ]


class PrototypeTransfer(nn.Module):
    def __init__(
        self,
        input_dim=64,
        num_classes=3,
        num_src_clusters=15,
        num_tgt_clusters=15,
        num_sources=14,
        src_momentum=0.5,
        tgt_momentum=0.5,
        use_pcw=False,
        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_sources = num_sources
        self.num_src_clusters = num_src_clusters
        self.num_tgt_clusters = num_tgt_clusters

        self.src_momentum = src_momentum
        self.tgt_momentum = tgt_momentum
        self.use_pcw = use_pcw

        
        self.register_buffer("src_cluster_labels", torch.zeros(num_sources, num_src_clusters, dtype=torch.long))
        self.register_buffer("src_cluster_centers", torch.zeros(num_sources, num_src_clusters, input_dim))
        self.register_buffer("tgt_cluster_centers", torch.zeros(num_tgt_clusters, input_dim))

        
        if self.use_pcw:
            self.register_buffer("src_cluster_confidence", torch.ones(num_sources, num_src_clusters))
            self.register_buffer("tgt_cluster_confidence", torch.ones(num_tgt_clusters))

        
        self.register_buffer("easy_select_counts", torch.ones(num_sources))
        self.momentum_min = 0.70
        self.momentum_max = 0.99

        self.extractor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim // 2),
            nn.Linear(input_dim // 2, input_dim),
        )

    
    
    

    def increment_easy_count(self, src_idx):
        self.easy_select_counts[src_idx] += 1

    def get_adaptive_momentum(self, src_idx):
        counts = self.easy_select_counts.float()
        freq_norm = (counts[src_idx] - counts.min()) / (counts.max() - counts.min() + 1e-6)
        momentum = self.momentum_min + (self.momentum_max - self.momentum_min) * freq_norm
        return momentum

    def get_source_vote_weights(self):
        counts = self.easy_select_counts.float()
        weights = counts.pow(0.5)
        weights = weights / (weights.sum() + 1e-8)
        return weights

    
    
    

    def compute_cluster_center(self, features, clusters, num_clusters):
        device = features.device
        clusters = clusters.long().to(device)
        one_hot = F.one_hot(clusters, num_classes=num_clusters).float()
        counts = one_hot.sum(dim=0) + 1e-6
        centers = torch.matmul(one_hot.T, features) / counts.unsqueeze(1)
        return centers

    def update_source_cluster_centers(self, features, clusters, idx):
        new_centers = self.compute_cluster_center(features, clusters, self.num_src_clusters)
        old_centers = self.src_cluster_centers[idx]
        momentum = self.get_adaptive_momentum(idx)
        self.src_cluster_centers[idx] = momentum * old_centers + (1 - momentum) * new_centers

        if self.use_pcw:
            new_conf = PrototypeConfidence.compute(features, clusters, self.num_src_clusters)
            self.src_cluster_confidence[idx] = (
                momentum * self.src_cluster_confidence[idx] + (1 - momentum) * new_conf
            )

    def update_target_cluster_centers(self, features, clusters):
        new_centers = self.compute_cluster_center(features, clusters, self.num_tgt_clusters)
        self.tgt_cluster_centers = (
            self.tgt_momentum * self.tgt_cluster_centers + (1 - self.tgt_momentum) * new_centers
        )

        if self.use_pcw:
            new_conf = PrototypeConfidence.compute(features, clusters, self.num_tgt_clusters)
            self.tgt_cluster_confidence = (
                self.tgt_momentum * self.tgt_cluster_confidence + (1 - self.tgt_momentum) * new_conf
            )

    
    
    

    def forward(self, src_feat, src_cluster, src_idx, tgt_feat, tgt_cluster):
        src_feat = self.extractor(src_feat)
        tgt_feat = self.extractor(tgt_feat)
        self.update_source_cluster_centers(src_feat, src_cluster, src_idx)
        tgt_cluster_label = self.map_target_clusters_to_labels(tgt_feat, tgt_cluster, src_idx)
        return tgt_cluster_label[tgt_cluster]

    
    
    

    @torch.no_grad()
    def map_target_clusters_to_labels(self, tgt_feat, tgt_cluster, src_idx):
        self.update_target_cluster_centers(tgt_feat, tgt_cluster)

        tgt_center = F.normalize(self.tgt_cluster_centers, dim=1)
        src_center = F.normalize(self.src_cluster_centers[src_idx], dim=1)
        sim_matrix = torch.matmul(tgt_center, src_center.T)

        if self.use_pcw:
            pcw_weight = (
                self.tgt_cluster_confidence.unsqueeze(1) *
                self.src_cluster_confidence[src_idx].unsqueeze(0)
            )
            sim_matrix = sim_matrix * pcw_weight

        nearest_src_cluster = torch.argmax(sim_matrix, dim=1)
        return self.src_cluster_labels[src_idx][nearest_src_cluster]

    
    
    

    @torch.no_grad()
    def predict(self, tgt_feat):
        tgt_feat = self.extractor(tgt_feat)
        tgt_centers = F.normalize(self.tgt_cluster_centers, dim=1)

        sim_tgt = torch.matmul(F.normalize(tgt_feat, dim=1), tgt_centers.T)
        tgt_cluster_idx = torch.argmax(sim_tgt, dim=1)

        domain_weights = self.get_source_vote_weights()
        vote_scores = torch.zeros(tgt_feat.size(0), self.num_classes, device=tgt_feat.device)

        for i in range(self.num_sources):
            src_centers = F.normalize(self.src_cluster_centers[i], dim=1)
            sim_src = torch.matmul(tgt_centers, src_centers.T)

            if self.use_pcw:
                pcw_weight = (
                    self.tgt_cluster_confidence.unsqueeze(1) *
                    self.src_cluster_confidence[i].unsqueeze(0)
                )
                sim_src = sim_src * pcw_weight

            nn_idx = torch.argmax(sim_src, dim=1)
            pred_labels = self.src_cluster_labels[i][nn_idx[tgt_cluster_idx]]

            vote_scores.scatter_add_(
                1,
                pred_labels.unsqueeze(1),
                torch.full((tgt_feat.size(0), 1), domain_weights[i], device=tgt_feat.device)
            )

        return torch.argmax(vote_scores, dim=1)

    
    
    

    @torch.no_grad()
    def initialize_cluster_labels(self, label_lists, cluster_lists):
        for i, (labels, clusters) in enumerate(zip(label_lists, cluster_lists)):
            true_labels = torch.argmax(labels, dim=-1)
            unique_clusters = np.unique(clusters)
            for cluster in unique_clusters:
                idx = np.where(clusters == cluster)[0]
                if len(idx) == 0:
                    continue
                majority = np.bincount(true_labels[idx]).argmax()
                self.src_cluster_labels[i, cluster] = majority

    @torch.no_grad()
    def initialize_cluster_centers(self, feat_lists, cluster_lists):
        for i, (feature, cluster) in enumerate(zip(feat_lists, cluster_lists)):
            feature = self.extractor(feature.cuda())
            centers = self.compute_cluster_center(feature, cluster.cuda(), self.num_src_clusters)
            self.src_cluster_centers[i] = centers

            if self.use_pcw:
                conf = PrototypeConfidence.compute(feature, cluster.cuda(), self.num_src_clusters)
                self.src_cluster_confidence[i] = conf

    def get_parameters(self):
        return [{"params": self.extractor.parameters(), "lr_mult": 1}]


class VEPRA(nn.Module):
    def __init__(
        self,
        input_dim=310,
        num_classes=3,
        max_iter=1000,
        transfer_loss_type="dann",
        num_src_clusters=15,
        num_tgt_clusters=15,
        num_sources=14,
        src_momentum=0.5,
        tgt_momentum=0.1,
        use_freq_attn=False,
        use_dynamic_fap=False,
        use_pcw=False,
        route_mode="reliability",
        route_seed=73171,
        route_cls_weight=0.7,
        route_sim_weight=0.3,
        cssc_gradient_scope="fc2",
        fap_alpha_init=0.1,
        fap_gate_hidden=64,
        fap_channel_reduction=4,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_sources = num_sources
        if route_mode not in {"reliability", "random", "same_high", "same_low"}:
            raise ValueError(f"Unknown route mode: {route_mode}")
        if cssc_gradient_scope not in {"fc2", "full"}:
            raise ValueError(f"Unknown CSSC gradient scope: {cssc_gradient_scope}")
        if route_cls_weight < 0 or route_sim_weight < 0:
            raise ValueError("Route score weights must be non-negative.")
        if route_cls_weight + route_sim_weight <= 0:
            raise ValueError("At least one route score weight must be positive.")
        self.route_mode = route_mode
        self.route_cls_weight = float(route_cls_weight)
        self.route_sim_weight = float(route_sim_weight)
        self.cssc_gradient_scope = cssc_gradient_scope
        self._route_generator = torch.Generator(device="cpu")
        self._route_generator.manual_seed(int(route_seed))

        self.shared_encoder = SharedEncoder(
            input_dim=input_dim,
            use_freq_attn=use_freq_attn,
            use_dynamic_fap=use_dynamic_fap,
            fap_alpha_init=fap_alpha_init,
            fap_gate_hidden=fap_gate_hidden,
            fap_channel_reduction=fap_channel_reduction,
        )
        self.emotion_classifier = EmotionClassifier(input_dim=64, num_classes=num_classes)
        self.prototype_transfer = PrototypeTransfer(
            input_dim=64, src_momentum=src_momentum, tgt_momentum=tgt_momentum,
            num_classes=num_classes, num_sources=num_sources,
            num_src_clusters=num_src_clusters, num_tgt_clusters=num_tgt_clusters,
            use_pcw=use_pcw
        )
        self.cls_loss = nn.CrossEntropyLoss()
        self.consis_loss = nn.CrossEntropyLoss()
        self.clu_loss = SupervisedContrastiveLoss(temperature=1)
        self.distribution_criterion = DomainAdaptationLoss(
            loss_type=transfer_loss_type, max_iter=max_iter, num_class=num_classes, **kwargs)

    def distribution_forward(self, src_feat, tgt_feat, src_label):
        src_logit = self.emotion_classifier(src_feat)
        loss_cls = self.cls_loss(src_logit, src_label)
        loss_adv = self.distribution_criterion(src_feat, tgt_feat)
        return loss_cls, loss_adv

    def prototype_forward(self, src_feat_easy, tgt_feat, src_cluster_easy, tgt_cluster):
        src_feat_easy_cpu = self.prototype_transfer.extractor(src_feat_easy).cpu()
        tgt_feat_cpu = self.prototype_transfer.extractor(tgt_feat).cpu()
        source_cluster_loss = self.clu_loss(src_feat_easy_cpu, src_cluster_easy.squeeze(0).cpu())
        target_cluster_loss = self.clu_loss(tgt_feat_cpu, tgt_cluster.squeeze(0).cpu())
        return source_cluster_loss, target_cluster_loss

    def forward(self, srcs, tgt, src_labels, src_clusters, tgt_cluster):
        srcs = srcs.permute(1, 0, 2)
        src_labels = src_labels.permute(1, 0, 2)
        src_clusters = src_clusters.permute(1, 0)

        distribution_source_idx, prototype_source_idx = self.route_sources(srcs, src_labels, tgt)
        self.prototype_transfer.increment_easy_count(prototype_source_idx)

        tgt_feat = self.shared_encoder(tgt)
        easy_feat = self.shared_encoder(srcs[prototype_source_idx])
        hard_feat = self.shared_encoder(srcs[distribution_source_idx])
        easy_cluster = src_clusters[prototype_source_idx]
        hard_label = src_labels[distribution_source_idx]

        loss_cls, loss_adv = self.distribution_forward(hard_feat, tgt_feat, hard_label)
        source_cluster_loss, target_cluster_loss = self.prototype_forward(easy_feat, tgt_feat, easy_cluster, tgt_cluster)

        tgt_proto_pred = self.prototype_transfer(
            easy_feat.detach(), easy_cluster, prototype_source_idx,
            tgt_feat.detach(), tgt_cluster
        )
        tgt_logit = self.emotion_classifier(tgt_feat)
        loss_consis = self.consis_loss(tgt_logit, tgt_proto_pred.to(tgt_logit.device))
        return loss_cls, loss_adv, loss_consis, source_cluster_loss, target_cluster_loss, prototype_source_idx, distribution_source_idx

    def predict(self, data, mode="target"):
        if mode == "source":
            return self.predict_by_hard(data)
        elif mode == "target":
            return self.predict_by_easy(data)

    def extract_cssc_feature(self, data):
        
        if self.cssc_gradient_scope == "fc2":
            feature = self.shared_encoder.forward_cssc_fc2_only(data)
        else:
            feature = self.shared_encoder(data)
        if feature.ndim != 2:
            raise RuntimeError(
                "Invalid VEPRA configuration."
            )
        return feature

    def predict_by_easy(self, data):
        self.eval()
        with torch.no_grad():
            feat = self.shared_encoder(data)
            pred = self.prototype_transfer.predict(feat)
        return pred

    def predict_by_hard(self, data):
        self.eval()
        with torch.no_grad():
            feat = self.shared_encoder(data)
            pred = self.emotion_classifier.predict(feat)
        return pred

    @torch.no_grad()
    def assess_source_reliability(self, srcs, src_labels, tgt):
        
        num_sources = srcs.size(0)
        scores = torch.zeros(num_sources, device=srcs.device)
        tgt_feat = self.shared_encoder(tgt)

        for i in range(num_sources):
            src_feat = self.shared_encoder(srcs[i])
            loss_cls, _ = self.distribution_forward(src_feat, tgt_feat, src_labels[i])

            sim = F.cosine_similarity(
                src_feat.mean(dim=0, keepdim=True),
                tgt_feat.mean(dim=0, keepdim=True),
                dim=1
            ).mean()

            scores[i] = (
                -self.route_cls_weight * loss_cls
                + self.route_sim_weight * sim
            )

        distribution_source_idx = torch.argmin(scores)
        prototype_source_idx = torch.argmax(scores)
        return distribution_source_idx, prototype_source_idx

    @torch.no_grad()
    def random_source_roles(self):
        if self.num_sources < 2:
            raise RuntimeError("Random routing requires at least two sources.")
        indices = torch.randperm(
            self.num_sources, generator=self._route_generator, device="cpu"
        )[:2]
        device = self.prototype_transfer.easy_select_counts.device
        distribution_source_idx, prototype_source_idx = indices.to(device=device, dtype=torch.long)
        return distribution_source_idx, prototype_source_idx

    @torch.no_grad()
    def route_sources(self, srcs, src_labels, tgt):
        if self.route_mode == "random":
            
            self.assess_source_reliability(srcs, src_labels, tgt)
            return self.random_source_roles()
        distribution_source_idx, prototype_source_idx = self.assess_source_reliability(srcs, src_labels, tgt)
        if self.route_mode == "same_high":
            distribution_source_idx = prototype_source_idx
        elif self.route_mode == "same_low":
            prototype_source_idx = distribution_source_idx
        return distribution_source_idx, prototype_source_idx

    def get_main_parameters(self):
        params = [*self.shared_encoder.get_parameters(), *self.emotion_classifier.get_parameters()]
        params.append({"params": self.distribution_criterion.loss_func.domain_classifier.parameters(), "lr_mult": 1})
        return params

    def get_prototype_parameters(self):
        return [*self.prototype_transfer.get_parameters()]

    @torch.no_grad()
    def on_training_start(self, srcs, src_clusters, src_labels):
        feats = [self.shared_encoder(src).cpu() for src in srcs]
        self.prototype_transfer.initialize_cluster_labels(src_labels, src_clusters)
        self.prototype_transfer.initialize_cluster_centers(feats, src_clusters)

    def get_state(self):
        proto = {
            "src_cluster_labels": self.prototype_transfer.src_cluster_labels.clone().detach(),
            "src_cluster_centers": self.prototype_transfer.src_cluster_centers.clone().detach(),
            "tgt_cluster_centers": self.prototype_transfer.tgt_cluster_centers.clone().detach(),
        }
        if self.prototype_transfer.use_pcw:
            proto["src_cluster_confidence"] = self.prototype_transfer.src_cluster_confidence.clone().detach()
            proto["tgt_cluster_confidence"] = self.prototype_transfer.tgt_cluster_confidence.clone().detach()
        return {
            "model": self.state_dict(),
            "proto": proto,
            "route_rng_state": self._route_generator.get_state().clone(),
        }

    def load_state(self, state):
        self.load_state_dict(state["model"])
        self.prototype_transfer.src_cluster_labels = state["proto"]["src_cluster_labels"]
        self.prototype_transfer.src_cluster_centers = state["proto"]["src_cluster_centers"]
        self.prototype_transfer.tgt_cluster_centers = state["proto"]["tgt_cluster_centers"]
        if "route_rng_state" in state:
            self._route_generator.set_state(state["route_rng_state"].cpu())
        if self.prototype_transfer.use_pcw and "src_cluster_confidence" in state["proto"]:
            self.prototype_transfer.src_cluster_confidence = state["proto"]["src_cluster_confidence"]
            self.prototype_transfer.tgt_cluster_confidence = state["proto"]["tgt_cluster_confidence"]
