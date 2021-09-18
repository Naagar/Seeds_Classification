# Seeds_Project

## Setup (Done only the first time)
* Clone the repo. 
* Download Miniconda from here --> https://docs.conda.io/en/latest/miniconda.html
* Once downloaded open the exectuable File 'Miniconda3-latest-Windows-x86_64'. And follow the usual installation process.
* After the installation gets completed open command-prompt and type conda --version. If you get a prompt saying: "conda 4.9.2", you have correctly installed.
* Create a conda env: 
     $ conda create -y -n at37 python=3.7

* Type 'conda activate at37' and Enter.
* Navigate to the local folder where you have the cloned repo.
* Ensure you are present in **Seeds_Project/Ann_Tool_Seeds_Proj**
* pip install -r requirements.txt

### Before executing the following 3 commands ensure all the required Image-Folders are present under the **Seeds_Project/Ann_Tool_Seeds_Proj/static** folder as subfolders.
#### Start of the very first session on a particular day. A session is said to end when you click on 'Stop Session' button in the Annotation Tool.
* python create_start_state.py --is_os_win 0 --initials hk --run 1 --global_reset 1 --img_dir_path ./static/Path2ImageFolder
* python create_start_state.py --is_os_win 0 --initials hk --run 1 --global_reset 0 --img_dir_path ./static/Path2ImageFolder
* python main_test_data_temp.py --is_os_win 0 --initials hk --img_dir_path ./static/Path2ImageFolder
* Copy everything after 'Dash is running on' say (http://127.0.0.1:7236) and open a new browser tab (say Chrome/Mozilla etc) and paste in the URL field of the tab.


#### Starting a session after the first session on a particular day.
* python create_start_state.py --is_os_win 0 --initials hk --run 1 --global_reset 0 --img_dir_path ./static/Path2ImageFolder
* python main_test_data_temp.py --is_os_win 0 --initials hk --img_dir_path ./static/Path2ImageFolder
* Copy everything after 'Dash is running on' say (http://127.0.0.1:7236) and open a new browser tab (say Chrome/Mozilla etc) and paste in the URL field of the tab.




