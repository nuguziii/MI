import logging
import time, os
from pathlib import Path
import torch

def create_logger(output_dir, description, phase='train'):
    root_output_dir = Path(output_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    final_output_dir = root_output_dir / description

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(description, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = final_output_dir / 'log'

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    if phase is 'test':
        result_dir = final_output_dir / 'result'
        result_dir.mkdir(parents=True, exist_ok=True)
        return logger, str(final_output_dir), str(result_dir)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def save_checkpoint(states, is_best, output_dir, model, epoch,
                    filename='checkpoint', epoch_size=20):
    if (epoch % epoch_size == 0):
        torch.save(model, os.path.join(output_dir, filename + '_' + str(epoch) + '.pth'))

    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'],
                   os.path.join(output_dir, 'model_best_state_dict.pth'))
        torch.save(model, os.path.join(output_dir, 'model_best.pth'))