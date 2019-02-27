# 25 February 2019 - Robert Li <robertwli@gmail.com>


import sys
from src.text_classifier_deprn_rates import DeprnPredictor


predict = DeprnPredictor()

print('Evaluate using user input.\n')
user_description = ['']
print('\"QQ\" to quit.')
print('\"CR\" to see classification report.')
print('Otherwise...')
while True:
    user_description = input('Enter a depreciable asset description: \n')
    if user_description == 'QQ':
        print('====================GOODBYE====================\n')
        sys.exit()
    elif user_description == 'CR':
        predict.report_results()
    else:
        result, predicted_account = predict.predict_description(user_description)
        rate_perc = str(result.rate_perc_text) + '% prime cost'
        life = str(result.life_years) + ' years effective life'
        tax_cat = result.tax_cat
        print(f'Input from user:\n\t {user_description}')
        print(f'Result:')
        print(f'\taccount: \t\t\t{predicted_account}')
        print(f'\tdeprn rate: \t\t{rate_perc}')
        print(f'\teffecrive life: \t{life}')
        print(f'\ttax category: \t\t{tax_cat}')
        print('END of Result')
        print()
