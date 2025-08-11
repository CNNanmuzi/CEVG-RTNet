import math
import torch
import torch.nn.functional as F

class IoU_Cal:
    ''' pred, target: x0,y0,x1,y1
        monotonous: {
            None: origin  v1
            True: monotonic FM v2
            False: non-monotonic FM  v3
        }
        momentum: The momentum of running mean (This can be set by the function <momentum_estimation>)'''
    iou_mean = 1.
    monotonous = None
    momentum = 1 - 0.5 ** (1 / 7000)
    _is_train = True

    @classmethod
    def momentum_estimation(cls, n, t):
        ''' n: Number of batches per training epoch
            t: The epoch when mAP's ascension slowed significantly'''
        time_to_real = n * t
        cls.momentum = 1 - pow(0.05, 1 / time_to_real)
        return cls.momentum

    def __init__(self, pred, target):
        self.pred, self.target = pred, target
        self._fget = {
            # x,y,w,h
            'pred_xy': lambda: (self.pred[..., :2] + self.pred[..., 2: 4]) / 2,
            'pred_wh': lambda: self.pred[..., 2: 4] - self.pred[..., :2],
            'target_xy': lambda: (self.target[..., :2] + self.target[..., 2: 4]) / 2,
            'target_wh': lambda: self.target[..., 2: 4] - self.target[..., :2],
            # x0,y0,x1,y1
            'min_coord': lambda: torch.minimum(self.pred[..., :4], self.target[..., :4]),
            'max_coord': lambda: torch.maximum(self.pred[..., :4], self.target[..., :4]),
            # The overlapping region
            'wh_inter': lambda: torch.relu(self.min_coord[..., 2: 4] - self.max_coord[..., :2]),
            's_inter': lambda: torch.prod(self.wh_inter, dim=-1),
            # The area covered
            's_union': lambda: torch.prod(self.pred_wh, dim=-1) +
                               torch.prod(self.target_wh, dim=-1) - self.s_inter,
            # The smallest enclosing box
            'wh_box': lambda: self.max_coord[..., 2: 4] - self.min_coord[..., :2],
            's_box': lambda: torch.prod(self.wh_box, dim=-1),
            'l2_box': lambda: torch.square(self.wh_box).sum(dim=-1),
            # The central points' connection of the bounding boxes
            'd_center': lambda: self.pred_xy - self.target_xy,
            'l2_center': lambda: torch.square(self.d_center).sum(dim=-1),
            # IoU
            'iou': lambda: self.s_inter / (self.s_union + 1e-8),
            
            # PolyIoU geometric quantities
            'pred_area': lambda: torch.prod(self.pred_wh, dim=-1),
            'target_area': lambda: torch.prod(self.target_wh, dim=-1),
            'pred_aspect': lambda: self.pred_wh[..., 0] / (self.pred_wh[..., 1] + 1e-8),
            'target_aspect': lambda: self.target_wh[..., 0] / (self.target_wh[..., 1] + 1e-8),
            'box_center': lambda: (self.min_coord[..., :2] + self.max_coord[..., 2:4]) / 2,
            'pred_center_offset': lambda: self.pred_xy - self.box_center,
            'target_center_offset': lambda: self.target_xy - self.box_center,
            'norm_pred_wh': lambda: self.pred_wh / (torch.sqrt(torch.prod(self.pred_wh, dim=-1, keepdim=True)) + 1e-8),
            'norm_target_wh': lambda: self.target_wh / (torch.sqrt(torch.prod(self.target_wh, dim=-1, keepdim=True)) + 1e-8),
        }
        self._update(self)

    def __setitem__(self, key, value):
        self._fget[key] = value

    def __getattr__(self, item):
        if callable(self._fget[item]):
            self._fget[item] = self._fget[item]()
        return self._fget[item]

    @classmethod
    def train(cls):
        cls._is_train = True

    @classmethod
    def eval(cls):
        cls._is_train = False

    @classmethod
    def _update(cls, self):
        if cls._is_train: cls.iou_mean = (1 - cls.momentum) * cls.iou_mean + \
                                         cls.momentum * self.iou.detach().mean().item()

    def _scaled_loss(self, loss, alpha=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            beta = self.iou.detach() / self.iou_mean
            if self.monotonous:
                loss *= beta.sqrt()
            else:
                divisor = delta * torch.pow(alpha, beta - delta)
                loss *= beta / divisor
        return loss

    def _dynamic_weights(self, distance_factor=1.0):
        iou_quality = 1 - self.iou.detach()
        center_diff = torch.sqrt(self.l2_center) / (torch.sqrt(self.l2_box) + 1e-8)
        aspect_diff = torch.abs(torch.log(self.pred_aspect + 1e-8) - torch.log(self.target_aspect + 1e-8))
        area_diff = torch.abs(torch.log(self.pred_area + 1e-8) - torch.log(self.target_area + 1e-8))
        
        w_center = torch.clamp(0.5 + center_diff * distance_factor, 0.1, 2.0)
        w_aspect = torch.clamp(0.3 + aspect_diff, 0.1, 1.5)
        w_area = torch.clamp(0.2 + area_diff, 0.1, 1.0)
        w_closure = torch.clamp(0.4 + iou_quality, 0.1, 1.2)
        w_overlap = torch.clamp(0.6 + center_diff, 0.2, 1.5)
        w_symmetry = torch.clamp(0.3 + aspect_diff * 0.5, 0.1, 0.8)
        
        return {
            'center': w_center,
            'aspect': w_aspect, 
            'area': w_area,
            'closure': w_closure,
            'overlap': w_overlap,
            'symmetry': w_symmetry
        }

    def _center_loss(self):
        return self.l2_center / (self.l2_box + 1e-8)

    def _aspect_loss(self):
        aspect_diff = torch.abs(self.pred_aspect - self.target_aspect)
        return aspect_diff / (torch.maximum(self.pred_aspect, self.target_aspect) + 1e-8)

    def _area_loss(self):
        area_diff = torch.abs(self.pred_area - self.target_area)
        return area_diff / (torch.maximum(self.pred_area, self.target_area) + 1e-8)

    def _closure_loss(self):
        return (self.s_box - self.s_union) / (self.s_box + 1e-8)

    def _weighted_overlap_loss(self, gamma=2.0):
        distance_weight = torch.exp(-gamma * torch.sqrt(self.l2_center) / (torch.sqrt(self.l2_box) + 1e-8))
        shape_similarity = torch.exp(-torch.abs(torch.log(self.pred_aspect + 1e-8) - torch.log(self.target_aspect + 1e-8)))
        weighted_iou_loss = (1 - self.iou) * distance_weight * shape_similarity
        return weighted_iou_loss

    def _symmetry_loss(self):
        center_symmetry = torch.abs(self.pred_center_offset - self.target_center_offset)
        center_sym_loss = torch.sum(center_symmetry, dim=-1) / (torch.sqrt(self.l2_box) + 1e-8)
        
        shape_symmetry = torch.abs(self.norm_pred_wh - self.norm_target_wh)
        shape_sym_loss = torch.sum(shape_symmetry, dim=-1)
        
        return (center_sym_loss + shape_sym_loss) / 2

    def _pca_shape_analysis(self):
        pred_features = torch.stack([self.pred_wh[..., 0], self.pred_wh[..., 1], 
                                   self.pred_aspect, torch.sqrt(self.pred_area)], dim=-1)
        target_features = torch.stack([self.target_wh[..., 0], self.target_wh[..., 1],
                                     self.target_aspect, torch.sqrt(self.target_area)], dim=-1)
        
        feature_diff = torch.abs(pred_features - target_features)
        weights = torch.tensor([0.3, 0.3, 0.25, 0.15], device=pred_features.device, dtype=pred_features.dtype)
        weighted_diff = torch.sum(feature_diff * weights, dim=-1)
        
        return weighted_diff

    @classmethod
    def IoU(cls, pred, target, self=None):
        self = self if self else cls(pred, target)
        return 1 - self.iou

    @classmethod
    def WIoU(cls, pred, target, self=None):
        self = self if self else cls(pred, target)
        dist = torch.exp(self.l2_center / (self.l2_box.detach() + 1e-8))
        return self._scaled_loss(dist * (1 - self.iou))

    @classmethod
    def EIoU(cls, pred, target, self=None):
        self = self if self else cls(pred, target)
        penalty = self.l2_center / (self.l2_box.detach() + 1e-8) \
                  + torch.square(self.d_center / (self.wh_box + 1e-8)).sum(dim=-1)
        return self._scaled_loss((1 - self.iou) + penalty)

    @classmethod
    def GIoU(cls, pred, target, self=None):
        self = self if self else cls(pred, target)
        return self._scaled_loss((1 - self.iou) + (self.s_box - self.s_union) / (self.s_box + 1e-8))

    @classmethod
    def DIoU(cls, pred, target, self=None):
        self = self if self else cls(pred, target)
        return self._scaled_loss((1 - self.iou) + self.l2_center / (self.l2_box + 1e-8))

    @classmethod
    def CIoU(cls, pred, target, eps=1e-4, self=None):
        self = self if self else cls(pred, target)
        v = 4 / math.pi ** 2 * \
            (torch.atan(self.pred_wh[..., 0] / (self.pred_wh[..., 1] + eps)) -
             torch.atan(self.target_wh[..., 0] / (self.target_wh[..., 1] + eps))) ** 2
        alpha = v / ((1 - self.iou) + v + 1e-8)
        return self._scaled_loss((1 - self.iou) + self.l2_center / (self.l2_box + 1e-8) + alpha.detach() * v)

    @classmethod
    def SIoU(cls, pred, target, theta=4, self=None):
        self = self if self else cls(pred, target)
        # Angle Cost
        angle = torch.arcsin(torch.abs(self.d_center).min(dim=-1)[0] / (self.l2_center.sqrt() + 1e-4))
        angle = torch.sin(2 * angle) - 2
        # Dist Cost
        dist = angle[..., None] * torch.square(self.d_center / (self.wh_box + 1e-8))
        dist = 2 - torch.exp(dist[..., 0]) - torch.exp(dist[..., 1])
        # Shape Cost
        d_shape = torch.abs(self.pred_wh - self.target_wh)
        big_shape = torch.maximum(self.pred_wh, self.target_wh)
        w_shape = 1 - torch.exp(- d_shape[..., 0] / (big_shape[..., 0] + 1e-8))
        h_shape = 1 - torch.exp(- d_shape[..., 1] / (big_shape[..., 1] + 1e-8))
        shape = w_shape ** theta + h_shape ** theta
        return self._scaled_loss((1 - self.iou) + (dist + shape) / 2)

    @classmethod
    def PolyIoU(cls, pred, target, distance_factor=1.0, gamma=2.0, 
                use_pca=True, traditional_weight=0.3, self=None):
        self = self if self else cls(pred, target)
        
        weights = self._dynamic_weights(distance_factor)
        
        center_loss = self._center_loss()
        aspect_loss = self._aspect_loss()
        area_loss = self._area_loss()
        closure_loss = self._closure_loss()
        overlap_loss = self._weighted_overlap_loss(gamma)
        symmetry_loss = self._symmetry_loss()
        
        poly_loss = (weights['center'] * center_loss +
                    weights['aspect'] * aspect_loss +
                    weights['area'] * area_loss +
                    weights['closure'] * closure_loss +
                    weights['overlap'] * overlap_loss +
                    weights['symmetry'] * symmetry_loss)
        
        if use_pca:
            pca_loss = self._pca_shape_analysis()
            poly_loss = poly_loss + 0.1 * pca_loss
        
        traditional_loss = 1 - self.iou
        final_loss = traditional_weight * traditional_loss + (1 - traditional_weight) * poly_loss
        
        return self._scaled_loss(final_loss)

    @classmethod
    def get_loss_components(cls, pred, target, distance_factor=1.0, gamma=2.0):
        self = cls(pred, target)
        weights = self._dynamic_weights(distance_factor)
        
        components = {
            'weights': weights,
            'losses': {
                'center': self._center_loss(),
                'aspect': self._aspect_loss(), 
                'area': self._area_loss(),
                'closure': self._closure_loss(),
                'overlap': self._weighted_overlap_loss(gamma),
                'symmetry': self._symmetry_loss(),
                'traditional_iou': self.iou
            },
            'geometric_info': {
                'pred_center': self.pred_xy,
                'target_center': self.target_xy,
                'pred_wh': self.pred_wh,
                'target_wh': self.target_wh,
                'intersection_area': self.s_inter,
                'union_area': self.s_union,
                'box_area': self.s_box
            }
        }
        
        return components