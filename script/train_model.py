
# -- add path --
import sys
sys.path.append("./lib/")

# -- imports --
from ustn import denoise_stn,default_config

def main():
    cfg = default_config()
    cfg.uuid = str(0)
    results = denoise_stn(cfg)
    print(results)

if __name__ == "__main__":
    main()
