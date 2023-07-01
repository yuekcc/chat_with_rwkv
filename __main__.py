import sys
import os

script_dir = os.path.dirname(__file__)

if __name__ == '__main__':
    sys.path.append(os.path.join(script_dir, '_vendor'))
    sys.path.append(os.path.join(script_dir, 'rwkv'))

    import app
    app.run()