"""
Created by huxo
Initialized @ 14:46, 2023/5/12
=========================================
This project is mainly about:
0.  machine learning tests on futures of equity index
"""

import os
import sys
import json
import platform

# platform confirmation
this_platform = platform.system().upper()
if this_platform == "WINDOWS":
    with open("/Deploy/config.json", "r") as j:
        global_config = json.load(j)
elif this_platform == "LINUX":
    with open("/home/huxo/Deploy/config.json", "r") as j:
        global_config = json.load(j)
else:
    print("... this platform is {}.".format(this_platform))
    print("... it is not a recognized platform, please check again.")
    sys.exit()

deploy_dir = global_config["deploy_dir"][this_platform]
project_data_root_dir = os.path.join(deploy_dir, "Data")

# --- calendar
calendar_dir = os.path.join(project_data_root_dir, global_config["calendar"]["calendar_save_dir"])
calendar_path = os.path.join(calendar_dir, global_config["calendar"]["calendar_save_file"])

# --- futures data
futures_dir = os.path.join(project_data_root_dir, global_config["futures"]["futures_save_dir"])
futures_shared_info_path = os.path.join(futures_dir, global_config["futures"]["futures_shared_info_file"])
futures_instru_info_path = os.path.join(futures_dir, global_config["futures"]["futures_instrument_info_file"])

futures_md_dir = os.path.join(futures_dir, global_config["futures"]["md_dir"])
futures_md_structure_path = os.path.join(futures_md_dir, global_config["futures"]["md_structure_file"])
futures_md_db_name = global_config["futures"]["md_db_name"]
futures_em01_db_name = global_config["futures"]["em01_db_name"]

futures_by_instrument_dir = os.path.join(futures_dir, global_config["futures"]["by_instrument_dir"])
major_minor_dir = os.path.join(futures_by_instrument_dir, global_config["futures"]["major_minor_dir"])
md_by_instru_dir = os.path.join(futures_by_instrument_dir, global_config["futures"]["md_by_instru_dir"])

# --- equity
equity_dir = os.path.join(project_data_root_dir, global_config["equity"]["equity_save_dir"])
equity_by_instrument_dir = os.path.join(equity_dir, global_config["equity"]["by_instrument_dir"])
equity_index_by_instrument_dir = os.path.join(equity_by_instrument_dir, global_config["equity"]["index_dir"])

# --- projects
projects_dir = os.path.join(deploy_dir, global_config["projects"]["projects_save_dir"])

# --- projects data
research_data_root_dir = "/ProjectsData"
research_project_name = os.getcwd().split("\\")[-1]
research_project_data_dir = os.path.join(research_data_root_dir, research_project_name)
research_features_and_return_dir = os.path.join(research_project_data_dir, "features_and_return")
research_ic_tests_dir = os.path.join(research_project_data_dir, "ic_tests")
research_ic_tests_summary_dir = os.path.join(research_project_data_dir, "ic_tests_summary")

if __name__ == "__main__":
    from skyrim.winterhold import check_and_mkdir

    check_and_mkdir(research_data_root_dir)
    check_and_mkdir(research_project_data_dir)
    check_and_mkdir(research_features_and_return_dir)
    check_and_mkdir(research_ic_tests_dir)
    check_and_mkdir(research_ic_tests_summary_dir)

    print("... directory system for this project has been established.")
