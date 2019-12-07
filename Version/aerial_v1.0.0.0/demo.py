import argparse
import os
from solver import SolverTest
from data_loader import get_testloader
from torch.backends import cudnn


def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net']:
        print('ERROR!! model_type should be selected in U_Net')
        print('Your input for model_type was %s'%config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)

    print(config)
    
    test_loader = get_testloader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)

    solver = SolverTest(config, test_loader)

    
    # Train and sample the images
    solver.test("./models/Aerial_Building.pkl", "./models/Aerial_Road.pkl")
    #solver.test("./models/Aerial_Building.pt", "./models/Aerial_Road.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0)
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=384)
    
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)

    # misc
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--model_type', type=str, default='U_Net', help='Only Support U_Net')
    parser.add_argument('--model_path', type=str, default='./models')

    # Aerial DataSet
    parser.add_argument('--test_path', type=str, default='./')

    parser.add_argument('--result_path', type=str, default='./result/')

    config = parser.parse_args()
    main(config)