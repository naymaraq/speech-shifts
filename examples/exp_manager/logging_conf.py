conf = """
version: 1
disable_existing_loggers: true
formatters:
  brief:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
handlers:
  console:
    level: INFO
    class: logging.StreamHandler
    formatter: brief
    stream : ext://sys.stdout
  file_handler:
    level: 'INFO'
    class: 'logging.handlers.WatchedFileHandler'
    formatter: 'brief'
    filename: './speech_shifts.log'
    mode: 'a'
    encoding: 'utf-8'
root:
  level: INFO
  handlers:
    - console
    - file_handler
"""