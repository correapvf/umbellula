
import os
import numpy as np
import pandas as pd
import cv2
from PySide6.QtCore import QThread

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# =========== Clean text and Interpolate ==============

def clean_text(series: pd.Series) -> pd.Series:
    replacements = str.maketrans({
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "D": "0",
        "C": "0",
        "A": "8",
        "B": "8",
        "S": "5",
    })

    return (
        series
        .fillna("")
        .str.translate(replacements)
        .str.strip()
        .str.replace(r"\D", "", regex=True)
    )

def clean_df(df, variable_string):
    # clear text
    cols = df.columns.to_list()
    cols = [col for col in cols if not col.endswith('_conf')]
    cols = [col for col in cols if col not in variable_string]
    cols = [col for col in cols if col not in ['video','timestamp_sec','image','folder']]

    df[cols] = df[cols].apply(clean_text)

    colsc =  [col for col in df.columns.to_list() if col.endswith('_conf')]
    df[colsc] = df[colsc].astype(float)

    for var in cols:
        if var.endswith('_dec'):
            var_int = var.split('_')[0]

            if var_int not in cols:
                continue

            df[var_int] = df[var_int] + '.' + df[var]

            var_intc = var_int + '_conf'
            varc = var + '_conf'
            df[var_intc] = df[[var_intc, varc]].min(axis=1)

            df = df.drop(columns=[var, varc])

    isdate = all(x in cols for x in ['day', 'month', 'year'])
    istime = all(x in cols for x in ['hour', 'minute', 'second'])
    if isdate and istime:
        df['datetime'] = (df['day'] + '/' + df['month'] + '/' + df['year'] + ' ' +
                          df['hour'] + ':' + df['minute'] + ':' + df['second'])
        df['datetime_conf'] = df[['day_conf', 'month_conf', 'year_conf', 'hour_conf', 'minute_conf', 'second_conf']].min(axis=1)

        df = df.drop(columns=['day', 'month', 'year', 'day_conf', 'month_conf', 'year_conf',
                              'hour', 'minute', 'second', 'hour_conf', 'minute_conf', 'second_conf'])
        isdate = istime = False

    if isdate:
        df['date'] = (df['day'] + '/' + df['month'] + '/' + df['year'])
        df['date_conf'] = df[['day_conf', 'month_conf', 'year_conf']].min(axis=1)

        df = df.drop(columns=['day', 'month', 'year', 'day_conf', 'month_conf', 'year_conf'])

    if istime:

        df['time'] = df['hour'] + ':' + df['minute'] + ':' + df['second']
        df['time_conf'] = df[['hour_conf', 'minute_conf', 'second_conf']].min(axis=1)

        df = df.drop(columns=['hour', 'minute', 'second', 'hour_conf', 'minute_conf', 'second_conf'])

    # Convert to numeric
    df_bak = df.copy()
    cols = ['latitude','longitude','depth','heading','altitude']
    cols = [col for col in cols if col in df]
    cols = [col for col in cols if col not in variable_string]

    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

    if 'datetime' in df:
        df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
        df['datetime'] = (df['datetime'] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        df.loc[df['datetime'] <= 0, 'datetime'] = np.nan

    if (df.isna().mean() > 0.5).any():
        # something went wrong, abort
        return df_bak

    # Interpolate
    if 'heading' in df:
        df['heading_cos'] = np.cos(np.radians(df['heading']))
        df['heading_sin'] = np.sin(np.radians(df['heading']))

    cols = ['latitude','longitude','depth','heading_cos','heading_sin','altitude','datetime']
    cols = [col for col in cols if col in df]
    cols = [col for col in cols if col not in variable_string]

    window = 15
    for col in cols:
        Q1 = df[col].rolling(window=window, center=True, min_periods=3).quantile(0.25)
        Q3 = df[col].rolling(window=window, center=True, min_periods=3).quantile(0.75)
        IQR = Q3 - Q1
        index = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
        df.loc[index, col] = np.nan

    df[cols] = df[cols].interpolate(limit = window)

    if 'datetime' in df:
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s').dt.round('s')

    if 'heading' in df:
        df['heading'] = np.degrees(np.atan2(df['heading_sin'], df['heading_cos'])) % 360
        df['heading'] = df['heading'].round(2)
        df = df.drop(columns=['heading_sin','heading_cos'])

    return df

# ================= Process video =================

def process_video(video_path, bboxes, output_csv, ocr, interval, 
                  parent_folder, progress2_signal):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open file {video_path}")

    if parent_folder:
        subfolders = video_path.relative_to(parent_folder).parts[:-1]
        subfolders = '/'.join(subfolders)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if frame_count > 0 else 0.0
    t = 0
    rows = []

    while t <= duration:

        if QThread.currentThread().isInterruptionRequested():
            break

        roi_frames_list = []

        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        success, frame = cap.read()

        if not success:
            break

        for var, (x1, y1, x2, y2) in bboxes.items():

            roi = frame[y1:y2, x1:x2]

            # roi = cv2.copyMakeBorder(
            #     roi,
            #     top=5,
            #     bottom=5,
            #     left=5,
            #     right=5,
            #     borderType=cv2.BORDER_CONSTANT,
            #     value=[0, 0, 0]
            # )

            roi_frames_list.append(roi)
        
        row = {
            "video": video_path.name,
            "timestamp_sec": t
        }

        if parent_folder:
            row['folder'] = subfolders

        result_list = ocr.predict(roi_frames_list)
        result_dict = dict(zip(bboxes.keys(), result_list))

        for var, result in result_dict.items():
            row[var] = result['rec_text']
            row[f"{var}_conf"] = round(result['rec_score'], 2)
       
        rows.append(row)

        t += interval
        progress2_signal.emit((t*100)//duration)

    if QThread.currentThread().isInterruptionRequested():
        return

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)


def main_video(video_files, bboxes, csv_output, interval, variable_string, clean_csv, 
               keep_tmp_csv, parent_folder, status, progress1_signal, progress2_signal):
    from paddleocr import TextRecognition
    ocr = TextRecognition()
    i = 1
    temp_csvs = []

    for video in video_files:

        if QThread.currentThread().isInterruptionRequested():
            break

        status.emit(f"Processing {video.name}")

        temp_csv = video.with_stem(video.stem + '_ocr_temp').with_suffix('.csv')

        # Pular se já existir
        if temp_csv.exists():
            temp_csvs.append(temp_csv)
            progress1_signal.emit(i)
            i += 1
            continue

        process_video(video, bboxes, temp_csv, ocr, interval, parent_folder, progress2_signal)
        temp_csvs.append(temp_csv)

        progress1_signal.emit(i)
        i += 1

    if QThread.currentThread().isInterruptionRequested():
        return
    # Concate CSVs
    status.emit("Genarating final output.")

    final_df = [pd.read_csv(csv, dtype=str) for csv in temp_csvs]

    if clean_csv:
        if keep_tmp_csv:
            raw_output = csv_output.with_stem(csv_output.stem + '_raw_ocr').with_suffix('.csv')
            raw_df = pd.concat(final_df, ignore_index=True)
            raw_df.to_csv(raw_output, index=False, encoding='utf-8-sig')

        final_df = [clean_df(df, variable_string) for df in final_df]

    final_df = pd.concat(final_df, ignore_index=True)
    final_df.to_csv(csv_output, index=False, encoding='utf-8-sig')

    for csv in temp_csvs:
        csv.unlink()

    return final_df

# ================= Process image =================

def main_image(img_files, bboxes, csv_output, variable_string, clean_csv, 
               parent_folder, keep_tmp_csv, status, progress1_signal):
    from paddleocr import TextRecognition
    ocr = TextRecognition()
    i = 1
    rows = []

    status.emit(f"Processing images...")

    for img in img_files:

        if QThread.currentThread().isInterruptionRequested():
            break

        frame = cv2.imread(str(img))

        if frame is None:
            continue

        row = {"image": img.name}

        if parent_folder:
            subfolders = img.relative_to(parent_folder).parts[:-1]
            row['folder'] = '/'.join(subfolders)

        roi_frames = []
        roi_order = []
        
        for var, (x1, y1, x2, y2) in bboxes.items():
            roi = frame[y1:y2, x1:x2]

            roi_frames.append(roi)
            roi_order.append(var)

        result_flat = ocr.predict(roi_frames)

        for var, result in zip(roi_order, result_flat):
            row[var] = result['rec_text']
            row[f"{var}_conf"] = round(result['rec_score'], 2)

        rows.append(row)

        progress1_signal.emit(i)
        i += 1

    if QThread.currentThread().isInterruptionRequested():
        return
    
    status.emit("Genarating final output.")

    df = pd.DataFrame(rows)

    if clean_csv:
        if keep_tmp_csv:
            raw_output = csv_output.with_stem(csv_output.stem + '_raw_ocr').with_suffix('.csv')
            df.to_csv(raw_output, index=False, encoding='utf-8-sig')

        df = clean_df(df, variable_string)

    df.to_csv(csv_output, index=False, encoding='utf-8-sig')

    return df
