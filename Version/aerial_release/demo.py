import argparse
import os
from solver import SolverTest
from data_loader import get_testloader
from torch.backends import cudnn


def main(config):
    cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
        
    test_loader = get_testloader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)

    solver = SolverTest(config, test_loader, "./models/Aerial_Building.pt", "./models/Aerial_Road.pt")

    solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0)
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=416)
    
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)


    # Aerial DataSet
    parser.add_argument('--test_path', type=str, default='./')

    parser.add_argument('--result_path', type=str, default='./result/')

    config = parser.parse_args()
    main(config)