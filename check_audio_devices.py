import sounddevice as sd

print("Available Audio Devices:")
print(sd.query_devices())

default_input = sd.default.device[0]
print(f"\nDefault Input Device Index: {default_input}")

try:
    dev_info = sd.query_devices(default_input, 'input')
    print(f"Default Device Info: {dev_info['name']}")
    print(f"Max Input Channels: {dev_info['max_input_channels']}")
    print(f"Default Samplerate: {dev_info['default_samplerate']}")
except Exception as e:
    print(f"Error querying default device: {e}")
