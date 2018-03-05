import numpy as np

LASER_NIS_FILE = 'nis_laser.txt'
LASER_CHI_PERCENTILES = {  # 2 dimensions
  5: 0.103,
  10: 0.211,
  90: 4.605,
  95: 5.991
}

RADAR_NIS_FILE = 'nis_radar.txt'
RADAR_CHI_PERCENTILES = {  # 3 dimensions
  5: 0.352,
  10: 0.584,
  90: 6.251,
  95: 7.815,
}

def print_stats(filename, desirable_percentiles):
  nis_values = np.loadtxt(filename)
  for q, q_desirable_val in desirable_percentiles.items():
    q_actual_val = np.percentile(nis_values, q)
    print("%ith percentile. Desirable: %f. Actual: %f." % (
          q, q_desirable_val, q_actual_val)) 

print("LASER")
print_stats(LASER_NIS_FILE, LASER_CHI_PERCENTILES)

print("\nRADAR")
print_stats(RADAR_NIS_FILE, RADAR_CHI_PERCENTILES)
