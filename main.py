from network_parser import parse
import logging
import argparse
import global_v as glv
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', action='store', dest='config',\
            help='The path of config file')
    parser.add_argument('-checkpoint', action='store', dest='checkpoint',\
            help='The path of checkpoint, if use checkpoint')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    if args.config is None:
        raise Exception('Unrecognized config file.')
    else:
        config_path = args.config

    logging.basicConfig(filename='result.log', level=logging.INFO)

    logging.info("start parsing settings")

    params = parse(config_path)

    logging.info("finish parsing settings")

    if torch.cuda.is_available():
        device = params['device']#torch.device("cuda")
        cuda.init()
        c_device = aboutCudaDevices()
        print(c_device.info())
        print("selected device: ", device)
    else:
        device = torch.device("cpu")
        print("No GPU is found")
    dtype = torch.float32

	glv.init(device, dtype, params)
