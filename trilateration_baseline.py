import re
import numpy as np
import pandas as pd
import scipy.optimize

last_estimate = []

def squared_diff(estimate, ap_values):
    total = 0.0
    for row in ap_values:
        ap_loc = np.array(row[:2])
        measured_dist = row[2]
        calc_dist = np.linalg.norm(ap_loc - estimate)
        total += (calc_dist - measured_dist) ** 2
    return total


def nls(df):
    global last_estimate

    ap_values = []
    for row in df.itertuples():
        ap_values.append([
            getattr(row, 'ap_loc_x'),
            getattr(row, 'ap_loc_y'),
            getattr(row, 'est_distance') / 100.0  # cm -> m
        ])

    ap_values.sort(key=lambda x: x[2])
    if len(ap_values) > 4:
        ap_values = ap_values[:4]

    ap_values = np.array(ap_values)
    start = ap_values[0, 0:2] if len(last_estimate) == 0 else last_estimate

    result = scipy.optimize.minimize(
        squared_diff,
        start,
        args=(ap_values,),
        method='L-BFGS-B',
        bounds=((0.0, 5.35), (0.0, 19.25)),  # room dimensions in meters
        options={'ftol': 1e-4, 'maxiter': 1e7}
    )

    last_estimate = result.x

    return pd.Series({
        'est_loc_x': result.x[0],
        'est_loc_y': result.x[1],
        'success': result.success
    })


def main():
    ap_coords = {   # fixed access points coordinates
        'ap1': (0.0, 0.0),
        'ap2': (5.35, 0.0),
        'ap3': (5.35, 9.5),
        'ap4': (5.35, 19.25),
        'ap5': (0.0, 19.25)
    }

    df_raw = pd.read_csv('office_measurements.csv', sep=';', decimal=',', index_col=0)

    results = []

    for col in df_raw.columns:
        rows = []
        for ap_name, distance in df_raw[col].items():
            if pd.isna(distance) or ap_name not in ap_coords:
                continue
            x, y = ap_coords[ap_name]
            rows.append({'ap_loc_x': x, 'ap_loc_y': y, 'est_distance': distance})

        if not rows:
            continue

        result = nls(pd.DataFrame(rows))

        true_x, true_y = 0.0, 0.0
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", col)
        if len(nums) >= 2:
            true_x, true_y = float(nums[0]), float(nums[1])

        results.append({
            'measurement_point': col,
            'true_x': true_x,
            'true_y': true_y,
            'est_x': round(result['est_loc_x'], 1),
            'est_y': round(result['est_loc_y'], 1),
        })

    if results:
        print(pd.DataFrame(results))


if __name__ == "__main__":
    main()