import math

def scale(in_low, in_high, out_low, out_high, val):
    return ((out_high - out_low) * (val - in_low)) /(in_high - in_low) + out_low
