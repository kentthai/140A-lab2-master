from pathlib import Path


def check_and_apply_weights(model, weights):
    model_dict = model.state_dict()
    for mlayer, wlayer in zip(model_dict, weights):
        msize = model_dict[mlayer].shape
        wsize = weights[wlayer].shape
        if msize != wsize:
            raise ValueError('Your model is incorrect!')
        model_dict[mlayer] = weights[wlayer]
    model.load_state_dict(model_dict)
    print('Your network seems correct!')


def check_collected_images(path='dataset'):
    path = Path(path)
    free = path/'free'
    blocked = path/'blocked'
    free_images = set(free.glob('*.jpg'))
    blocked_images = set(blocked.glob('*.jpg'))
    if len(free_images) < 50:
        print(f'You only have {len(free_images)} unique `free` images. Collect more!')
    if len(blocked_images) < 50:
        print(f'You only have {len(blocked_images)} unique `blocked` images. Collect more!')
    if len(free_images) >= 50 and len(blocked_images) >= 50:
        print('You are good to go!')


if __name__ == '__main__':
    check_collected_images()