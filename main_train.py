from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
import datetime
import SinGAN.Augment as Augment

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='./MarbleData')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    opt = functions.post_config(opt)    # 인자값 opt에 저장
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    opt.input_name = opt.input_name
    dir2save = functions.generate_dir2save(opt)

    opt.scaleList = [0.8,0.8,0.8,0.8,0.75,0.75,0.75,0.65,0.65,0.65,0.65,0.65] #fixed
    opt.train_stages = len(opt.scaleList)
    opt.continue_traing = 'N'
    opt.cont = 'N'
    if (os.path.exists(dir2save)):
        opt.cont = input("Trained model exist. Do you want to train the model again? N/Y ")

    if(opt.cont == 'Y') :
        starting_point = int(input("From what scale do you want to start training?"))
        opt.starting_point = starting_point
        opt.continue_traing = 'Y'

    if (os.path.exists(dir2save) and (opt.cont == 'N')):
        print('trained model already exist')
    
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        
        # img_path = []
        # img_path.append(opt.input_dir)
        # img_path.append(opt.input_name)
        # img_path = '/'.join(img_path)
        #aug_real = Augment.Augment(img_path, opt)  
        #functions.adjust_scales2image(aug_real, opt)

        real = functions.read_image(opt)       
        functions.adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, NoiseAmp)
        SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)

       
