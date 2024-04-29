# logging setup

import logging
import logging.handlers
import sys

# Change root logger level from WARNING (default) to NOTSET in order for all messages to be delegated.
logging.getLogger().setLevel(logging.NOTSET)

console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# Add file rotating handler, with level DEBUG
rotatingHandler = logging.handlers.RotatingFileHandler(filename='rotating.log', maxBytes=10485760, backupCount=20, encoding='utf-8')
rotatingHandler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
rotatingHandler.setFormatter(formatter)
logging.getLogger().addHandler(rotatingHandler)

# Add file rotating handler, with level INFO for clef.* modules only
rotatingHandler_clef_info = logging.handlers.RotatingFileHandler(filename='clef-info.log', maxBytes=10485760, backupCount=20, encoding='utf-8')
rotatingHandler_clef_info.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
rotatingHandler_clef_info.setFormatter(formatter)
rotatingHandler_clef_info.addFilter(logging.Filter("clef"))
logging.getLogger().addHandler(rotatingHandler_clef_info)


# filter out lower level messages from certain modules
logger_descope=logging.getLogger('kivy.jnius')
logger_descope.setLevel(logging.WARN)
logger_descope=logging.getLogger('httpcore.http11')
logger_descope.setLevel(logging.WARN)
logger_descope=logging.getLogger('httpx')
logger_descope.setLevel(logging.WARN)
logger_descope=logging.getLogger('openai._base_client')
logger_descope.setLevel(logging.INFO)
