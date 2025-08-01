from configparser import ConfigParser

# Function to read the database configuration file
def config(filename='interview_db.ini', section='postgresql'):
    parser = ConfigParser()
    parser.read(filename)

    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
            # Convert port to integer since psycopg2 requires it
            if param[0] == 'port':
                db[param[0]] = int(param[1])
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db
