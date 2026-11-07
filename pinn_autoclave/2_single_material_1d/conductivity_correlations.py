

temperature_input = 600

def k_inc_value(
        T
):

    return (
        6.9995
        + 0.0151
        * T
    )


def k_nh3_value(
        T
):


    return (
        0.3039497375
        - 0.00019097
        * T
        + 0.0000001887
        * T
    )

print (k_inc_value(600))
print (k_nh3_value(600))
