# -*- coding: utf-8 -*-
"""
| **@created on:** 9/11/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| 
|
| **Sphinx Documentation Status:** 
"""

import psycopg2
import argparse
import glob
import datetime


class RawImages(object):
    def __init__(self, file_path:str):
        self.experiment = file_path.split("/")[-4]
        self.fileName = file_path.split("/")[-1]
        tmp_split = self.fileName.split("_")
        self.epoch = tmp_split[2]
        self.step = tmp_split[4]
        self.batch = tmp_split[6]
        self.evalImageId = tmp_split[-1].split(".")[0]
        self.uniqueId = f"{self.experiment}_E{self.epoch}_S{self.step}_B{self.batch}_EID{self.evalImageId}"
        self.image = psycopg2.Binary(open(file_path, 'rb').read())
        self.timestamp = datetime.datetime.now()


def insert_record(conn, rawImage: RawImages):
    try:
        cursor = conn.cursor()
        cursor.execute(
            f'INSERT INTO raw_images(experiment, epoch, step, batch, "uniqueId", image, timestamp, "fileName", "evalImageId") ' +
            f"VALUES('{rawImage.experiment}', '{rawImage.epoch}', '{rawImage.step}', '{rawImage.batch}', "
            f"'{rawImage.uniqueId}', {rawImage.image}, '{rawImage.timestamp}', '{rawImage.fileName}', "
            f"'{rawImage.evalImageId}')")
        conn.commit()
        cursor.close()
        print(f"{rawImage.fileName} inserted successfully")
    except Exception as e:
        print(e)
        exit()

def insert_options(conn, option):
    try:
        cursor = conn.cursor()
        cursor.execute(
            f'INSERT INTO "filter_options"(option) ' +
            f"VALUES('{option}')")
        conn.commit()
        cursor.close()
        print(f"{option} inserted successfully")
    except Exception as e:
        print(e)
        exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--path', type=str, help='Experiment Path',
                        default="/Users/prathyushsp/concept_learning_old/Exp_08_032_128_10/prediction_results/reconstructed_images")
    args = parser.parse_args()

    conn = psycopg2.connect(host="localhost",
                            port="5432",
                            user="postgres",
                            password="clearn@123",
                            database="app")
    files = list(glob.glob(args.path + "/*"))
    fl = len(files)

    filter_options = set()
    for e,file in enumerate(files):
        print(f"Working on {e}/{fl} [{round(e/fl*100,2)}%]")
        rm = RawImages(file)
        filter_options.add(rm.experiment)
        filter_options.add(f"{rm.experiment}_E{rm.epoch}")
        filter_options.add(f"{rm.experiment}_E{rm.epoch}_S{rm.step}")
        filter_options.add(f"{rm.experiment}_E{rm.epoch}_S{rm.step}_B{rm.batch}")
        insert_record(conn, RawImages(file))

    for option in filter_options:
        insert_options(conn, option)

    print("Experiment inserted successfully!")
