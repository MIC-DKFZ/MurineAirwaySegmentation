from typing import Tuple

import numpy as np
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.channel_selection_transforms import (
    DataChannelSelectionTransform,
    SegChannelSelectionTransform,
)
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    BrightnessTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.local_transforms import (
    BrightnessGradientAdditiveTransform,
    LocalGammaTransform,
)
from batchgenerators.transforms.noise_transforms import (
    BlankRectangleTransform,
    GaussianBlurTransform,
    GaussianNoiseTransform,
    MedianFilterTransform,
    SharpeningTransform,
)
from batchgenerators.transforms.resample_transforms import (
    SimulateLowResolutionTransform,
)
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    Rot90Transform,
    SpatialTransform,
    TransposeAxesTransform,
)
from batchgenerators.transforms.utility_transforms import (
    NumpyToTensor,
    RemoveLabelTransform,
    RenameTransform,
)
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.custom_transforms import (
    Convert2DTo3DTransform,
    Convert3DTo2DTransform,
    ConvertSegmentationToRegionsTransform,
    MaskTransform,
)
from nnunet.training.data_augmentation.default_data_augmentation import (
    default_3D_augmentation_params,
)
from nnunet.training.data_augmentation.downsampling import (
    DownsampleSegForDSTransform2,
    DownsampleSegForDSTransform3,
)
from nnunet.training.data_augmentation.pyramid_augmentations import (
    ApplyRandomBinaryOperatorTransform,
    MoveSegAsOneHotToData,
    RemoveRandomConnectedComponentFromOneHotEncodingTransform,
)
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from scipy.ndimage import gaussian_filter
from torch import nn


class InhomogeneousSliceIlluminationTransform(AbstractTransform):
    """
    This transform simulates inhomogeneous illumination across image slices, introducing intensity variations
    to mimic realistic imaging artifacts.

    Attributes:
        num_defects (Tuple): Range for the number of illumination defects to introduce.
        defect_width (Tuple): Range for the width of the defects.
        mult_brightness_reduction_at_defect (Float): Brightness reduction at defect areas.
        base_p (tuple): Base probability for defects to appear.
        base_red (Tuple[float, float]): Range of reduction factors for defect intensities.
        p_per_sample (float): Probability of applying the transform per image sample.
        per_channel (bool): Whether to apply the transform independently per channel.
        p_per_channel (float): Probability of applying the transform to each channel.
        data_key (str): Key for accessing the data within the data dictionary.
    """
    def __init__(self, num_defects, defect_width, mult_brightness_reduction_at_defect, base_p,
                 base_red: Tuple[float, float], p_per_sample=1, per_channel=True, p_per_channel=0.5, data_key='data'):
        super().__init__()
        self.num_defects = num_defects
        self.defect_width = defect_width
        self.mult_brightness_reduction_at_defect = mult_brightness_reduction_at_defect
        self.base_p = base_p
        self.base_red = base_red
        self.p_per_sample = p_per_sample
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.data_key = data_key

    @staticmethod
    def _sample(stuff):
        """
        Samples a value based on its type.

        Args:
            value: Can be a float, int, tuple, list, or a callable function.

        Returns:
            A sampled value based on the type of the input.
        """
        if isinstance(stuff, (float, int)):
            return stuff
        elif isinstance(stuff, (tuple, list)):
            assert len(stuff) == 2
            return np.random.uniform(*stuff)
        elif callable(stuff):
            return stuff()
        else:
            raise ValueError('Invalid input for sampling.')

    def _build_defects(self, num_slices):
        """
        Constructs the inhomogeneous illumination pattern by creating Gaussian-shaped defects.

        Args:
            num_slices (int): Number of slices in the 3D image.

        Returns:
            np.ndarray: Array of intensity factors for each slice.
        """
        int_factors = np.ones(num_slices)

        # gaussian shaped ilumination changes
        num_gaussians = int(np.round(self._sample(self.num_defects)))
        for n in range(num_gaussians):
            sigma = self._sample(self.defect_width)
            pos = np.random.choice(num_slices)
            tmp = np.zeros(num_slices)
            tmp[pos] = 1
            tmp = gaussian_filter(tmp, sigma, mode='constant', truncate=3)
            tmp = tmp / tmp.max()
            strength = self._sample(self.mult_brightness_reduction_at_defect)
            int_factors *= (1 - (tmp * (1 - strength)))
        int_factors = np.clip(int_factors, 0.1, 1)
        ps = np.ones(num_slices) / num_slices
        ps += (1 - int_factors) / num_slices  # probability in defect areas is twice as high as in the rest
        ps /= ps.sum()
        idx = np.random.choice(num_slices, int(np.round(self._sample(self.base_p) * num_slices)), replace=False, p=ps)
        noise = np.random.uniform(*self.base_red, size=len(idx))
        int_factors[idx] *= noise
        int_factors = np.clip(int_factors, 0.1, 2)
        return int_factors

    def __call__(self, **data_dict):
        """
        Applies the inhomogeneous illumination transform to the input data.

        Args:
            data_dict (dict): Dictionary containing the input data.

        Returns:
            dict: Transformed data dictionary.
        """
        data = data_dict.get(self.data_key)
        assert data is not None
        assert len(
            data.shape) == 5, "this only works on 3d images, the provided tensor is 4d, so it's a 2d image (bcxy)"
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                if self.per_channel:
                    for c in range(data.shape[1]):
                        if np.random.uniform() < self.p_per_channel:
                            defects = self._build_defects(data.shape[2])
                            data[b, c] *= defects[:, None, None]
                else:
                    defects = self._build_defects(data.shape[2])
                    for c in range(data.shape[1]):
                        if np.random.uniform() < self.p_per_channel:
                            data[b, c] *= defects[:, None, None]
        data_dict[self.data_key] = data
        return data_dict


def get_moreDA_augmentation_airway(dataloader_train, dataloader_val, patch_size, params=default_3D_augmentation_params,
                                   border_val_seg=-1,
                                   seeds_train=None, seeds_val=None, order_seg=1, order_data=3,
                                   deep_supervision_scales=None,
                                   soft_ds=False,
                                   classes=None, pin_memory=True, regions=None,
                                   use_nondetMultiThreadedAugmenter: bool = False):
    """
    Creates data augmentation pipelines for training and validation datasets with extended transformations specific
    to airway segmentation tasks.

    Args:
        dataloader_train: Dataloader for training data.
        dataloader_val: Dataloader for validation data.
        patch_size: The size of the patches to be used for spatial transformations.
        params: Dictionary containing parameters for augmentations.
        border_val_seg: Value to be used for segmentation borders.
        seeds_train: Seeds for randomization in training.
        seeds_val: Seeds for randomization in validation.
        order_seg: Order of interpolation for segmentation data.
        order_data: Order of interpolation for image data.
        deep_supervision_scales: Scales for deep supervision during training.
        soft_ds: Whether to use soft deep supervision.
        classes: List of classes for segmentation.
        pin_memory: Whether to use pinned memory for data loaders.
        regions: Segmentation regions to convert to regions.
        use_nondetMultiThreadedAugmenter: Flag to use a non-deterministic augmenter.

    Returns:
        Tuple: Augmenters for training and validation data.
    """
    assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"

    tr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        ignore_axes = (0,)
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size
        ignore_axes = None

    tr_transforms.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None,
        do_elastic_deform=params.get("do_elastic"), alpha=params.get("elastic_deform_alpha"),
        sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), p_rot_per_axis=params.get("rotation_p_per_axis"),
        do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg,
        order_seg=order_seg, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))

    if params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    tr_transforms.append(
        InhomogeneousSliceIlluminationTransform(
            (1, 5),
            (2, 8),
            lambda: np.random.uniform(0.2, 0.6) if np.random.uniform() < 0.8 else np.random.uniform(0.7, 1.2),
            (0, 0.3),
            (0.25, 2),
            0.3,
            False,
            1,
            'data'
        )
    )

    if params.get("rot90"):
        # print('using rot90')
        matching_axes = np.array([sum([i == j for j in patch_size]) for i in patch_size])
        if np.any(matching_axes > 1):
            valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])
            tr_transforms.append(
                Rot90Transform(
                    (0, 1, 2, 3), axes=valid_axes, data_key='data', label_key='seg', p_per_sample=0.3
                ),
            )

    if params.get("transpose_axes"):
        # print('using transpose_axes')
        matching_axes = np.array([sum([i == j for j in patch_size]) for i in patch_size])
        if np.any(matching_axes > 1):
            valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])
            tr_transforms.append(
                TransposeAxesTransform(valid_axes, data_key='data', label_key='seg', p_per_sample=0.5)
            )

    # we need to put the color augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color
    # channel gets in the way
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))

    if params.get("do_additive_brightness"):
        tr_transforms.append(BrightnessTransform(params.get("additive_brightness_mu"),
                                                 params.get("additive_brightness_sigma"),
                                                 True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                                 p_per_channel=params.get("additive_brightness_p_per_channel")))

    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=ignore_axes))
    tr_transforms.append(
        GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
                       p_per_sample=0.1))  # inverted gamma

    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"]))

    if params.get("do_mirror") or params.get("mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    if params.get("blank_rectangles"):
        # print('using blank_rectangles')
        tr_transforms.append(
            BlankRectangleTransform([[max(1, p // 10), p // 3] for p in patch_size],
                                    rectangle_value=np.mean,
                                    num_rectangles=(1, 5),
                                    force_square=False,
                                    p_per_sample=0.4,
                                    p_per_channel=0.5)
        )

    if params.get("do_gaussian_int_grad"):
        tr_transforms.append(
            BrightnessGradientAdditiveTransform(
                lambda x, y: np.random.uniform(x[y] // 8, x[y] // 2),
                (-0.5, 1.5),
                max_strength=lambda x, y: np.random.uniform(-3, -1) if np.random.uniform() < 0.5 else np.random.uniform(
                    1, 3),
                same_for_all_channels=False,
                p_per_sample=0.2,
                p_per_channel=0.5
            )
        )

    if params.get("do_local_gamma"):
        tr_transforms.append(
            LocalGammaTransform(
                lambda x, y: np.random.uniform(x[y] // 8, x[y] // 2),
                lambda x, y: np.random.uniform(-0.5, 0.5) if np.random.uniform() < 0.5 else np.random.uniform(0.5, 1.5),
                lambda: np.random.uniform(0.01, 0.8) if np.random.uniform() < 0.5 else np.random.uniform(1.5, 4),
                same_for_all_channels=False,
                p_per_sample=0.2,
                p_per_channel=0.5
            )
        )

    if params.get("do_median_filter"):
        tr_transforms.append(
            MedianFilterTransform(
                (1, 5),
                same_for_each_channel=False,
                p_per_sample=0.2,
                p_per_channel=0.5
            )
        )

    if params.get("do_sharpening"):
        tr_transforms.append(
            SharpeningTransform(
                strength=(0.1, 1),
                same_for_each_channel=False,
                p_per_sample=0.2,
                p_per_channel=0.5
            )
        )

    if params.get("mask_was_used_for_normalization") is not None:
        mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
        tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        tr_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
        if params.get("cascade_do_cascade_augmentations") is not None and params.get(
                "cascade_do_cascade_augmentations"):
            if params.get("cascade_random_binary_transform_p") > 0:
                tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                    p_per_sample=params.get("cascade_random_binary_transform_p"),
                    key="data",
                    strel_size=params.get("cascade_random_binary_transform_size"),
                    p_per_label=params.get("cascade_random_binary_transform_p_per_label")))
            if params.get("cascade_remove_conn_comp_p") > 0:
                tr_transforms.append(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                        key="data",
                        p_per_sample=params.get("cascade_remove_conn_comp_p"),
                        fill_with_other_class_p=params.get("cascade_remove_conn_comp_max_size_percent_threshold"),
                        dont_do_if_covers_more_than_X_percent=params.get(
                            "cascade_remove_conn_comp_fill_with_other_class_p")))

    tr_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        tr_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            tr_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
        else:
            tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                              output_key='target'))

    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    tr_transforms = Compose(tr_transforms)

    if use_nondetMultiThreadedAugmenter:
        if NonDetMultiThreadedAugmenter is None:
            raise RuntimeError('NonDetMultiThreadedAugmenter is not yet available')
        batchgenerator_train = NonDetMultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                            params.get("num_cached_per_thread"), seeds=seeds_train,
                                                            pin_memory=pin_memory)
    else:
        batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                      params.get("num_cached_per_thread"),
                                                      seeds=seeds_train, pin_memory=pin_memory)
    # batchgenerator_train = SingleThreadedAugmenter(dataloader_train, tr_transforms)
    # import IPython;IPython.embed()

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))

    val_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            val_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
        else:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                               output_key='target'))

    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    if use_nondetMultiThreadedAugmenter:
        if NonDetMultiThreadedAugmenter is None:
            raise RuntimeError('NonDetMultiThreadedAugmenter is not yet available')
        batchgenerator_val = NonDetMultiThreadedAugmenter(dataloader_val, val_transforms,
                                                          max(params.get('num_threads') // 2, 1),
                                                          params.get("num_cached_per_thread"),
                                                          seeds=seeds_val, pin_memory=pin_memory)
    else:
        batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms,
                                                    max(params.get('num_threads') // 2, 1),
                                                    params.get("num_cached_per_thread"),
                                                    seeds=seeds_val, pin_memory=pin_memory)
    # batchgenerator_val = SingleThreadedAugmenter(dataloader_val, val_transforms)
    return batchgenerator_train, batchgenerator_val


class nnUNetTrainerV2_airwayAug(nnUNetTrainerV2):
    """
    Custom trainer class that extends nnUNetTrainerV2 to include specialized data augmentation techniques for airway
    segmentation.
    """
    def initialize(self, training=True, force_load_plans=False):
        """
        relative to nnUNetTrainerV2 all we do here is replace the original data augmentation scheme with
        get_moreDA_augmentation_airway. The rest is unchanged

        Args:
            training (bool): If True, initializes the training data generators.
            force_load_plans (bool): If True, forces the loading of plan files.
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation_airway(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=True
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True
