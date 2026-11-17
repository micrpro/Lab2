import time
from mcp3021_driver import MCP3021
import adc_plot

dr = 5.2
du = 30

def main():
    adc = None
    voltage_values = []
    time_values = []
    try:
        adc = MCP3021(dr)
        start = time.time()
        while time.time()-start < du:
            voltage_values.append(adc.get_voltage())
            time_values.append(time.time() - start)
    finally:
        if adc is not None:
            adc.deinit()

    if time_values:
        adc_plot.plot_voltage_vs_time(time_values, voltage_values, dr)
        with open('values.csv', 'w', newline='') as f:
            for a, b in zip(time_values, voltage_values):
                f.write(f"{a}, {b}\n")

if __name__ == '__main__':
    main()