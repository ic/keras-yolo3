import os

import matplotlib

# in case you are running on machine without display, e.g. server
if not os.environ.get('DISPLAY', '') and matplotlib.rcParams['backend'] != 'agg':
    print('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

# If you want to force non-interactive mode (e.g. when using VirtualEnv)
if os.environ.get('FORCE_NON_INTERACTIVE', ''):
    print('Force using non-interactive Agg backend')
    matplotlib.use('Agg')
