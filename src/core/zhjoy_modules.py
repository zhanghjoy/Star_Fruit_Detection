######################################## Efficient Up-Convolution Block (EUCB) ########################################
class EUCB(nn.Module):
    """
    Efficient Up-Convolution Block (EUCB)
    This module is designed to perform efficient feature upsampling and channel interaction.
    It combines depthwise convolution with pixel shuffle operations to reduce computation cost
    while enhancing feature representation after upsampling.
    """
    def __init__(self, in_channels, kernel_size=3, stride=1):
        super(EUCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels

        # Depthwise Convolution after upsampling
        # 1. Upsample the input feature map by a factor of 2
        # 2. Apply depthwise convolution (grouped by input channels)
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(self.in_channels, self.in_channels, kernel_size, g=self.in_channels, s=stride)
        )

        # Pointwise Convolution (1x1 Conv)
        # Used for channel mixing after depthwise operations
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def channel_shuffle(self, x, groups):
        """
        Channel Shuffle operation.
        Redistributes channels across groups to promote cross-channel information exchange.
        This improves feature diversity without adding extra parameters.
        """
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups

        # Reshape: [B, C, H, W] -> [B, G, C/G, H, W]
        x = x.view(batchsize, groups, channels_per_group, height, width)
        # Transpose channel groups
        x = torch.transpose(x, 1, 2).contiguous()
        # Flatten back to [B, C, H, W]
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        """
        Forward propagation of EUCB.
        1. Upsample and apply depthwise convolution
        2. Shuffle channels for better feature interaction
        3. Apply pointwise convolution to fuse features
        """
        x = self.up_dwc(x)
        x = self.channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x
######################################## Efficient Up-Convolution Block (EUCB) ########################################


######################################## Efficient Detect Head (EDH) ########################################
class EDH(nn.Module):
    """
    Efficient Detect Head (EDH)
    A lightweight detection head for object detection tasks.
    - Uses grouped convolutions for feature extraction (reduces FLOPs).
    - Predicts bounding box offsets and class scores simultaneously.
    - Supports dynamic anchor generation and distribution focal loss (DFL).
    """
    dynamic = False  # flag for dynamic grid reconstruction
    export = False   # flag for export mode
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=()):
        """
        Args:
            nc (int): number of object classes
            ch (tuple): input channel dimensions from different feature maps
        """
        super().__init__()
        self.nc = nc                     # number of classes
        self.nl = len(ch)                # number of detection layers
        self.reg_max = 16                # number of bins for distance prediction
        self.no = nc + self.reg_max * 4  # output channels (cls + bbox distribution)
        self.stride = torch.zeros(self.nl)

        # Stem: feature refinement using two group convolutions
        self.stem = nn.ModuleList(
            nn.Sequential(
                Conv(x, x, 3, g=x // 16),
                Conv(x, x, 3, g=x // 16)
            ) for x in ch
        )

        # Regression branch: predict bounding box distribution
        self.cv2 = nn.ModuleList(nn.Conv2d(x, 4 * self.reg_max, 1) for x in ch)
        # Classification branch: predict class probabilities
        self.cv3 = nn.ModuleList(nn.Conv2d(x, self.nc, 1) for x in ch)

        # Distribution Focal Loss transformation
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """
        Forward pass of EDH.
        During training: outputs raw regression and classification predictions.
        During inference: generates decoded bounding boxes and class probabilities.
        """
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = self.stem[i](x[i])
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            # Update anchors and strides dynamically
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # Concatenate feature maps across all scales
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        # Split into box and class predictions
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):
            box = x_cat[:, :self.reg_max * 4]  # avoid TensorFlow ops mismatch
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        # Decode bounding boxes and apply sigmoid to class scores
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """
        Initialize biases for classification and regression heads.
        Helps stabilize training in early epochs.
        """
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a.bias.data[:] = 1.0  # regression bias
            b.bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # classification bias
######################################## Efficient Detect Head (EDH) ########################################

######################################## Multi-Scale Edge Enhancer (MSEE) ########################################
class EdgeEnhancer(nn.Module):
    """
    EdgeEnhancer
    A lightweight module for enhancing edge information in feature maps.
    - Uses average pooling to extract low-frequency components
    - Subtracts pooled features from original input to obtain edge details
    - Applies a convolution + sigmoid activation to enhance edge responses
    """
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        """
        Forward pass of EdgeEnhancer.
        1. Extract local smooth region using AvgPool
        2. Compute edge = input - pooled(input)
        3. Enhance edges through convolution + sigmoid
        4. Add enhanced edges back to original input
        """
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge


class MSEE(nn.Module):
    """
    Multi-Scale Edge Enhancer (MSEE)
    A feature enhancement module that integrates edge-aware learning with multi-scale context.
    - Extracts features at multiple spatial scales via adaptive pooling
    - Enhances edges at each scale using EdgeEnhancer
    - Fuses local and multi-scale features for robust representation
    """
    def __init__(self, inc, bins):
        """
        Args:
            inc (int): input channel dimension
            bins (list): list of pooling sizes for multi-scale feature extraction
        """
        super().__init__()
        self.features = []
        for bin in bins:
            # Multi-scale feature extraction:
            # 1. Adaptive pooling to (bin x bin)
            # 2. Channel reduction via 1x1 convolution
            # 3. Grouped 3x3 convolution for spatial refinement
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                Conv(inc, inc // len(bins), 1),
                Conv(inc // len(bins), inc // len(bins), 3, g=inc // len(bins))
            ))

        # Edge enhancers for each scale
        self.ees = []
        for _ in bins:
            self.ees.append(EdgeEnhancer(inc // len(bins)))

        self.features = nn.ModuleList(self.features)
        self.ees = nn.ModuleList(self.ees)

        # Local convolution (context feature extraction)
        self.local_conv = Conv(inc, inc, 3)
        # Final fusion convolution
        self.final_conv = Conv(inc * 2, inc)

    def forward(self, x):
        """
        Forward pass of MSEE.
        1. Extract local features using 3x3 convolution
        2. For each scale:
            - Perform adaptive pooling
            - Apply edge enhancement
            - Upsample back to original size
        3. Concatenate local and multi-scale enhanced features
        4. Fuse via final convolution
        """
        x_size = x.size()
        out = [self.local_conv(x)]
        for idx, f in enumerate(self.features):
            out.append(
                self.ees[idx](F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
            )
        return self.final_conv(torch.cat(out, 1))


class C3k_MSEE(C3k):
    """
    C3k_MSEE
    A C3k block integrated with MSEE module for edge-aware feature refinement.
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(MSEE(c_, [3, 6, 9, 12]) for _ in range(n)))


class C3k2_MSEE(C3k2):
    """
    C3k2_MSEE
    A C3k2 block integrated with MSEE module for multi-scale edge enhancement.
    Supports stacking of multiple MSEE modules.
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_MSEE(self.c, self.c, 2, shortcut, g) if c3k else MSEE(self.c, [3, 6, 9, 12])
            for _ in range(n)
        )
######################################## Multi-Scale Edge Enhancer (MSEE) ########################################

######################################## Pinwheel-shaped Convolution (PSConv) ########################################
class PSConv(nn.Module):
    """
    Pinwheel-shaped Convolution (PSConv)
    A convolutional operator designed to capture directional features more effectively.
    - Splits convolution into four pinwheel-like branches (rotated receptive fields)
    - Each branch captures horizontal/vertical edge patterns using asymmetric kernels
    - Concatenates directional features and fuses them via 2x2 convolution
    This enhances representation for irregular-shaped objects.
    """
    def __init__(self, c1, c2, k, s):
        """
        Args:
            c1 (int): input channels
            c2 (int): output channels
            k (int): kernel size for asymmetric convolution
            s (int): stride
        """
        super().__init__()
        # Define padding for four pinwheel orientations
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]

        # Convolutions with asymmetric kernels
        # cw: captures vertical edges with (1, k) kernels
        # ch: captures horizontal edges with (k, 1) kernels
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)

        # Final fusion convolution (2x2 kernel)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        """
        Forward pass of PSConv.
        1. Apply four asymmetric convolutions with pinwheel-like padding
        2. Concatenate the four directional features
        3. Fuse concatenated features via 2x2 convolution
        """
        yw0 = self.cw(self.pad[0](x))  # vertical branch 1
        yw1 = self.cw(self.pad[1](x))  # vertical branch 2
        yh0 = self.ch(self.pad[2](x))  # horizontal branch 1
        yh1 = self.ch(self.pad[3](x))  # horizontal branch 2
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))
######################################## Pinwheel-shaped Convolution (PSConv) ########################################
