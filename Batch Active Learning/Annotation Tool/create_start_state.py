import json  
from glob import glob
import argparse
from datetime import date
import os



today = date.today()




# =========CLI Parsing===============
arg_container = argparse.ArgumentParser(description='Specify the Operating System')

# should be optional arguments container.add_arguments
arg_container.add_argument('--is_os_win', '-Is_Operating_System_Windows', type=int, required=True,
                           help='Is your OS Windows? type "--is_os_win 1" else 0')

arg_container.add_argument('--initials', "-Annotator's_name_initials", type=str, required=True,
                           help='example: if your name is Prateek Pani, type "--initials pp"')

arg_container.add_argument('--run', '-Run', type=bool, required=True,
                           help='Do you want to run this file? (--run 1) else 0')

arg_container.add_argument('--global_reset', '-global_reset', type=int, required=True,
                           help='Do you want to start from scratch? (--global_reset 1) else start_state only for today then --global_reset 0')

arg_container.add_argument('--img_dir_path', '-img_dir_path', type=str, required=True,
                           help='Enter the path of the image_folder')

# container.parse_args() and store in args
args = arg_container.parse_args()
# ====================================



def create_folder(today, name_initials):
    '''Should create a folder if not present and if present print'''
    day, month, year = today.day, today.month, today.year
    if args.is_os_win == 0:
        path = './StatsIO/{}/{}_{}_{}'.format(name_initials, day, month, year)
    else:
        path = '.\\StatsIO\\{}\\{}_{}_{}'.format(name_initials, day, month, year)
    try:
        os.makedirs(path)
        print("Directory created successfully")
    except OSError as error:
        print("Directory already present")

name_initials = args.initials
create_folder(today, str(name_initials))




def reset_json_file():
    '''When r'''
    day, month, year = today.day, today.month, today.year

    paths_of_images = glob(f'{args.img_dir_path}/*.png')
    pool_idx_list = list(range(len(paths_of_images)))
    hk_list = []
    hk_dict = {}

    hk_list = [idx for idx in range(len(paths_of_images))]
    encodings = ['0', '1', '2', '3'] #meanings ==> '0->broken'....'3->silkcut'  based on folder_name
    file_names = [paths_of_images[i].split('/')[-1] for i in range(len(paths_of_images))]
    print(file_names[0])

    # print(img[i])

    # comment Line-74 and uncomment Line-75
    # hk_dict = {f"img_{int(le)}.png": int(le)%5 for le in hk_list}
    hk_dict = {f'{file_names[i]}':encodings[0] for i in range(len(file_names))}


    time_log_dict = {"time_logs": []}

    # non-win OS
    if args.is_os_win == 0:

        if args.initials == 'hk':


            #reads the unseen_idx space dont update
            if args.global_reset == 1:
                with open(f'your_file_{args.initials}.txt', 'w') as f:
                    for item in hk_list:
                        f.write("%s\n" % item)



            with open(f"StatsIO/{args.initials}/{day}_{month}_{year}/yest_inp_file.json", "w") as fp:
                json.dump(obj=hk_dict, fp=fp)

            session_dict = {'session_num': 1}
            with open(f"StatsIO/{args.initials}/{day}_{month}_{year}/session_num.json", "w") as fp:
                json.dump(obj=session_dict, fp=fp)

            file1 = open(f"last_checkpoint_{args.initials}.txt", "w")
            file1.write('0')
            file1.close()

            with open(f"StatsIO/{args.initials}/{day}_{month}_{year}/time_logs.json", "w") as fp:
                fp.write(json.dumps(time_log_dict))

            with open(f"StatsIO/{args.initials}/{day}_{month}_{year}/images_annotated_today.json", "w") as fp:
                fp.write(json.dumps({}))



# ------------------------
    # win OS
    else:
        if args.initials == 'hk':


            if args.global_reset == 1:
                with open('your_file_hk.txt', 'w') as f:
                    for item in hk_list:
                        f.write("%s\n" % item)


            with open(f"StatsI\\hk\\{day}_{month}_{year}\\yest_inp_file.json", "w") as fp:
                json.dump(obj=hk_dict, fp=fp)


            session_dict = {'session_num': 1}
            with open(f"StatsIO\\{args.initials}\\{day}_{month}_{year}\\session_num.json", "w") as fp:
                json.dump(obj=session_dict, fp=fp)

            file1 = open(f"last_checkpoint_{args.initials}.txt", "w")
            file1.write('0')
            file1.close()

            with open(f"StatsIO\\{args.initials}\\{day}_{month}_{year}\\time_logs.json", "w") as fp:
                fp.write(json.dumps(time_log_dict))

            with open(f"StatsIO\\{args.initials}\\{day}_{month}_{year}\\images_annotated_today.json", "w") as fp:
                fp.write(json.dumps({}))


   
    print('\nSuccessfully reset the json file!! Can Start parsing again from scratch\n')
    return 

# uncomment when hard-reset
if args.run == 1:
    reset_json_file()



# json.loads