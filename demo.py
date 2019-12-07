import argparse
import os
from solver import SolverTest
from data_loader import get_testloader
from torch.backends import cudnn
import random

USE_Invoice = False

def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    print(config)
    
    test_loader = get_testloader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)

    solver = SolverTest(config, test_loader)

    
    # Train and sample the images
    if USE_Invoice:
        #solver.test("/home/root1/Learning/Image_Segmentation/models/Invoice_R2AttU_Net-250-0.0001-51-0.1398.pkl")
        solver.test("/home/root1/Learning/Image_Segmentation/models/Invoice_R2AttU_Net-150-0.0001-116-0.6092.pkl")
    else:
        #solver.test("/home/root1/Learning/Image_Segmentation/models/Aerial_R2AttU_Net-200-0.0002-39-0.4505.pkl")
        #solver.test("/home/root1/Learning/Image_Segmentation/models/Aerial_U_Net-250-0.0004-72-0.0531.pkl")
        solver.test("/home/root1/Learning/Image_Segmentation/models/Aerial_building_dice_loss_U_Net-250-0.0005-17-0.0022.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=7)
    parser.add_argument('--use_crf', type=bool, default=False)
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')

    # misc
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='./models')
    #parser.add_argument('--train_path', type=str, default='./dataset/train/')
    #parser.add_argument('--vaild_path', type=str, default='./dataset/vaild/')
    #parser.add_argument('--test_path', type=str, default='./dataset/test/')
    if USE_Invoice:
        # Invoice DataSet
        parser.add_argument('--train_path', type=str, default='/home/root1/dataSet_wwl/segmentation/Invoice/train/')
        parser.add_argument('--vaild_path', type=str, default='/home/root1/dataSet_wwl/segmentation/Invoice/val/')
        #parser.add_argument('--test_path', type=str, default='/home/root1/dataSet_wwl/segmentation/Invoice/test/')
        parser.add_argument('--test_path', type=str, default='/home/root1/Learning/Image_Segmentation/')
    else:
        # Aerial DataSet
        parser.add_argument('--train_path', type=str, default='/home/root1/dataSet_wwl/segmentation/Aerial_building/train/')
        parser.add_argument('--vaild_path', type=str, default='/home/root1/dataSet_wwl/segmentation/Aerial_building/val/')
        parser.add_argument('--test_path', type=str, default='/home/root1/dataSet_wwl/segmentation/Aerial_building/test/')
        #parser.add_argument('--test_path', type=str, default='/home/root1/Learning/Image_Segmentation/')

#     # IDCard DataSet
#     parser.add_argument('--train_path', type=str, default='/home/root1/dataSet_wwl/segmentation/IDCard/train/')
#     parser.add_argument('--vaild_path', type=str, default='/home/root1/dataSet_wwl/segmentation/IDCard/val/')
#     parser.add_argument('--test_path', type=str, default='./images/')


    parser.add_argument('--result_path', type=str, default='./result/')

    config = parser.parse_args()
    main(config)