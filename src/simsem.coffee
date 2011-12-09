###
CoffeeScript for demonstrating SimSem on its webpage.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2012-12-09
###

# Live example
EXAMPLE_TEXT = '''
Examples:
matrix metalloprotease-9
Tumor
patient
3,6-di(2,3-epoxypropoxy)xanthone
endothelial cell responsiveness to both VEGF and hypoxia
Try it out live...
'''

class LiveInput
    constructor: (@id) ->
        $live_input = $ @id
        $live_input.val EXAMPLE_TEXT

        # Clear the textbox on the first click
        $live_input.focus (event) ->
            $live_input.val ''
            $live_input.unbind event

class SimSem
    run: () ->
        new LiveInput '#live_input'

# Attach ourselves to the global context
root = exports ? this
root.SimSem = SimSem
