#
# Created on Sat Feb 03 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Problem: It is not easy to start Ray dashboard on windows due to the symbolic link
# Purpose: Support opening Ray dashboard on Windows
# Test Env: Windows 11- AMD64
# Tested libs: Ray 2.9.1, grafana-enterprise-10.3.1.windows-amd64, prometheus-2.49.1.windows-amd64.zip
# Extra libs: typed-argument-parser, pyyaml
# Tutorial: 
#       1. Start your ray program first.
#       2. Open another cmd or git bash
#       3. Run the below command
#           python start_dashboard.py --grafana_home_path <your_grafana_home_path> --prometheus_home_path <your_prometheus_home_path> 
#       4. For convenience, change the default paths in the below DashBoardConfig class
#       5. You can access Ray dashboard/ Prometheus/ Grafana 
#       
# ------------------------------
#
from pathlib import Path
import os
import subprocess
import time
import logging
from typing import Literal
import logging
import threading
from tap import Tap
import yaml
from configparser import ConfigParser

# Configure the overall logging
logging.basicConfig(level=logging.INFO)

class DashBoardConfig(Tap):
    ray_path:str   = ''                                                                             # by default we use the latest ray temp path (/tmp/ray/) or you define it here
    start_strategy:Literal['both','promo','grafana'] ='both'                                        # start prometheus server, grafana server, or both
    override_config: bool = True                                                                    # prometheus: open .yml and change latest path to a concrete latest path | grafana: open .ini and change to a concrete latest path 
    grafana_home_path: str=r"D:\Program Files\GrafanaLabs\grafana"                                  # path to grafana folder
    prometheus_home_path: str =r"D:\Program Files\prometheus"                                       # path to prometheus folder

def create_logger(name:str)-> logging.Logger:
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)  # Set the minimum logging level for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - ') 
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def output_reader(proc, name):
    logger = logging.getLogger(name)
    try:
        if proc.stdout:
            for line in iter(proc.stdout.readline, b''):
                logger.info(line.strip())
        if proc.stderr:
            for line in iter(proc.stderr.readline, b''):
                logger.error(line.strip())
    except KeyboardInterrupt:
        pass

def get_latest_session_path(ray_path:str =None)->str:
    home_path = str(Path.home())
    #home_path = home_path.replace('\\','/') 
    ray_path = os.path.join(home_path,'AppData','Local','Temp','ray')   #f'{home_path}/AppData/Local/Temp/ray' 
    dir_list =  sorted([ray_dir_name  for ray_dir_name in os.listdir(ray_path)  if os.path.isdir(os.path.join(ray_path, ray_dir_name))])
    latest_session = dir_list[-1]
    latest_path = os.path.join(ray_path, latest_session)
    return latest_path



def start_prometheus(latest_ray_path: str, prometheus_home_path="D:\Program Files\prometheus", override_config:bool= True) -> subprocess.Popen: #
    prometheus_folder = os.path.join(latest_ray_path,'metrics','prometheus') #f'{latest_ray_path}/metrics/prometheus'
    prometheus_yml_file = os.path.join(prometheus_folder, 'prometheus.yml') #f'{prometheus_path}/prometheus.yml'

    prometheus_exec_path = os.path.join(prometheus_home_path,'prometheus.exe')
    if override_config:
        prom_json_path =  os.path.join(str(Path.home()),'AppData','Local','Temp','ray','prom_metrics_service_discovery.json') 
        with open(prometheus_yml_file, 'r') as file:
            config = yaml.safe_load(file)
        config['scrape_configs'][0]['file_sd_configs'][0]['files'] = [ prom_json_path]
        with open(prometheus_yml_file, 'w') as file:
            yaml.dump(config, file)
    

    prometheus_yml_file = prometheus_yml_file.replace('\\','/')

    process = subprocess.Popen([prometheus_exec_path,'--config.file',prometheus_yml_file], stdout=subprocess.PIPE, shell=True)

    return process

def start_grafana(latest_ray_path: str, grafana_home_path:str= r"D:\Program Files\GrafanaLabs\grafana" , override_config:bool= True) -> subprocess.Popen:
    grafana_folder = os.path.join(latest_ray_path,'metrics','grafana') #f'{latest_ray_path}/metrics/prometheus'
    grafana_ini_file = os.path.join(grafana_folder, 'grafana.ini') #f'{grafana_folder}/grafana.ini'
    grafana_exec_path =  os.path.join(grafana_home_path,'bin','grafana-server.exe') #r"D:\Program Files\GrafanaLabs\grafana\bin\grafana-server.exe"

    os.chdir(os.path.join(grafana_home_path,'bin'))
    full_command = [grafana_exec_path, '--config', grafana_ini_file, '--homepath', grafana_home_path]
    process = subprocess.Popen(full_command,  stdout=subprocess.PIPE,  shell=True,)
    
    if override_config:
        new_provisioning_path = os.path.join(grafana_folder, 'provisioning')
        config = ConfigParser()
        config.read(grafana_ini_file)
        config['paths']['provisioning']=new_provisioning_path
        # Write the modified configuration back to the file
        with open(grafana_ini_file, 'w') as config_file:
            config.write(config_file)

    
    return process


if __name__=='__main__':

    config = DashBoardConfig()
    config.parse_args()
    start_strategy: Literal['promo','grafana','both'] = config.start_strategy
    latest_ray_path = get_latest_session_path(None) if (config.ray_path is None or config.ray_path == '' ) else config.ray_path
    
    # Create a logger
    logger = create_logger(name=__name__)

    logger.info(f'Detected the latest (custom) path: {latest_ray_path}')
    logger.info(f'start_strategy = {start_strategy}')


    prometheus_process = grafana_process= None
    try:
        process_list = []
        if start_strategy in ['both','promo']:
            prometheus_process = start_prometheus(latest_ray_path=latest_ray_path,
                                                  prometheus_home_path=config.prometheus_home_path,
                                                  override_config= config.override_config,
                                                  )
            
            if prometheus_process:
                logger.info('#'*80)
                logger.info(f'PROMETHEUS IS STARTING...')
                

        if start_strategy in ['both','grafana']:
            grafana_process = start_grafana(latest_ray_path=latest_ray_path,
                                            grafana_home_path= config.grafana_home_path,
                                            override_config=config.override_config,
                                            )
            if grafana_process:
                logger.info('#'*80)
                logger.info(f'GRAFANA IS STARTING...')
                t2 = threading.Thread(target=output_reader, args=(grafana_process, 'GRAFANA'))
                t2.start()
                    

        
        while(True):
            time.sleep(1)

    except KeyboardInterrupt:
        if prometheus_process:
            prometheus_process.kill()
        if grafana_process:
            t2.join(timeout=0.1)
            grafana_process.kill()
    except Exception as e:
        print(e)
    finally:
        exit(0)
