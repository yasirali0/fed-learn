import logging
from Server import Server
from options import args_parser
if __name__ == "__main__":

    args = args_parser()

    # Set logging
    logging.basicConfig(
        format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()),
        datefmt='%H:%M:%S')

    logging.info("log:{}".format(args.log))

    logger_file = logging.getLogger("File")
    logger_file.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("log/" + args.file_name, mode="w")
    formatter = logging.Formatter(' %(name)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger_file.addHandler(stream_handler)
    logger_file.addHandler(file_handler)

    # load config

    # server = Server(config)
    server = Server(args, logger_file)
    server.run()
