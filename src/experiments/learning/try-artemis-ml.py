from artemis.experiments import experiment_function

@experiment_function
def multiply_3_numbers(a=1, b=2, c=3):
    print("trocadero")
    return a*b*c

if __name__ == '__main__':
    multiply_3_numbers.add_variant('higher-ab', a=4, b=5)
    #record = multiply_3_numbers.run()
    ex = multiply_3_numbers.get_variant('higher-ab')
    ex.run()
    multiply_3_numbers.browse()