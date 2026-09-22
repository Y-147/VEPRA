

import torch


class PrototypeConfidence(object):

    @staticmethod
    def compute(
        features,
        clusters,
        num_clusters
    ):
        

        device = features.device

        confidence = []

        clusters = clusters.long().to(device)

        for c in range(num_clusters):

            mask = (clusters == c)

            
            if mask.sum() == 0:

                confidence.append(
                    torch.tensor(
                        0.1,
                        device=device
                    )
                )

                continue

            feat_c = features[mask]

            variance = feat_c.var(
                dim=0,
                unbiased=False
            ).mean()

            conf = torch.exp(
                -variance
            )

            conf = torch.clamp(
                conf,
                min=0.1,
                max=1.0
            )

            confidence.append(conf)

        return torch.stack(
            confidence
        )