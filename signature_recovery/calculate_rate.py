import re

log_file_path = 'signature_recovery/output_logs/recover_weights_layer_0.log'

try:
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    total_clusters = len(re.findall(r'CLUSTER ID', content))
    successful_recoveries = len(re.findall(r'Extracted weight vector \[-', content))
    failed_recoveries = len(re.findall(r'Extracted weight vector None', content))

    print(f"Inspecting log file: {log_file_path}")
    print("---")
    print(f"Total clusters attempted: {total_clusters}")
    print(f"Successful weight recoveries: {successful_recoveries}")
    print(f"Failed weight recoveries: {failed_recoveries}")
    print("---")

    if total_clusters > 0:
        success_rate = (successful_recoveries / total_clusters) * 100
        print(f"Success Rate: {success_rate:.2f}%")
    else:
        print("No clusters found to calculate a success rate.")

except FileNotFoundError:
    print(f"Error: Log file not found at {log_file_path}")
except Exception as e:
    print(f"An error occurred: {e}") 