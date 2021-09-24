## STEPS TO COMPLETE ##
# 1. import logging
# 2. set up config file for logging called `results.log`
# 3. add try except with logging for success or error
#    in relation to checking the types of a and b
# 4. check to see that log is created and populated correctly
#    should have error for first function and success for
#    the second
# 5. use pylint and autopep8 to make changes
#    and receive 10/10 pep8 score

import logging 

logging.basicConfig(
    filename ='./results.log',
    level =logging.INFO,
    filemode ='w',
    format = '%name(s) - %levelname(s), %message(s)')

def sum_vals(a, b):
    '''
    Args:
        a: (int)
        b: (int)
    Return:
        a + b (int)
    '''
    try:
        if a == int(a):
            logging.info("SUCCESS: Number a is integer")
    except:
        logging.error("Error: Number {} is not an integer".format(a))
    try:
        if b == int(b):
            logging.info("SUCCESS: Number {} is integer")
    except:
        logging.error("Error: Number {} is not an integer".format(b))  
        
    return a+b

if __name__ == "__main__":
    sum_vals('no', 'way')
    sum_vals(4, 5)
