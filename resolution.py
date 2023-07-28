from realesrgan import RealESRGANer, RRDBNet

def resolution(input_img):
    """
    Inference demo for Real-ESRGAN.
    """
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4

    # determine model paths
    model_path = 'models/RealESRGAN_x4plus.pth'

    # use dni to control the denoise strength
    dni_weight = None

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device='cuda',
        gpu_id=0)

    try:
        output, _ = upsampler.enhance(input_img, outscale=4)
        return output
    except RuntimeError as error:
        return f'Error {error}\nIf you encounter CUDA out of memory, try to set tile with a smaller number.'