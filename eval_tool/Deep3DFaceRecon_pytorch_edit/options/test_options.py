"""This script contains the test options for Deep3DFaceRecon_pytorch_edit
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--dataset_mode', type=str, default=None, help='chooses how datasets are loaded. [None | flist]')
        parser.add_argument('--img_folder', type=str, default='examples', help='folder for test images.')

        parser.add_argument('--batch-size', type=int, default=1,
                            help='Batch size to use')
        parser.add_argument('--num-workers', type=int,
                            help=('Number of processes to use for data loading. '
                                'Defaults to `min(8, num_cpus)`'))
        parser.add_argument('--device', type=str, default=None,
                            help='Device to use. Like cuda, cuda:0 or cpu')
        parser.add_argument('--mask', type=bool, default=True,
                            help='whether to use mask or not')
        # parser.add_argument('--dims', type=int, default=2048,
        #                     choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
        #                     help=('Dimensionality of Inception features to use. '
        #                           'By default, uses pool3 features'))
        # parser.add_argument('--path', type=str, nargs=2,
        #                     default=['dataset/FaceData/CelebAMask-HQ/CelebA-HQ-img', 'results/test_bench/results'],
        #                     help=('Paths to the generated images or '
        #                         'to .npz statistic files'))

        # parser.add_argument('--print_sim', type=bool, default=False,)

        # Dropout and Batchnorm has different behavior during training and test.
        self.isTrain = False
        return parser
