model_name = "mistralai/Mistral-7B-Instruct-v0.2"
device = "cuda"

text = """Text:\nAs I peered out the window this morning, the skies above presented a palette of grays, with clouds amassing as if in preparation for a grand display. The air felt crisp, hinting at the onset of what could be either a refreshing breeze or a chilling wind. Leaves on the trees fluttered slightly, indicating that the wind was already beginning its day's work. One might wonder whether these signs foretell a day of gentle showers or perhaps a stern downpour. With such a spectacle unfolding, a person can't help but ask oneself the essential question that guides our choice of apparel and activities for the day: What is the weather like today? Is it a day for warm layers and boots, or might a light jacket suffice?"""

standard_instruct = """Summarize the text:\n"""

standard_summary = """Summary:\nWhat is the weather like today?"""

shakespearean_instruct = """Think step by step. Step 1, summarize the text and output a summary. Step 2, convert the summary to Shakespearean English\n"""

shakespearean_summary = """Summary:\nWhat is the weather like today?"\nShakespearean Summary:\nVerily, what manner of tempest doth grace the skies this day?"""

test_text = """Text:\nAndy Murray tied the knot with Kim Sears on Saturday at Dunblane Cathedral, as thousands of fans waited outside to greet the couple. The tennis star, who announced his engagement in November last year, tied the knot in the 12th century cathedral before a relatively small group of their family members and close friends. Reverend Colin Renwick, led the service, was pictured outside the cathedral just after 3.00pm this afternoon with the wedding taking place at around 4.00pm. Andy Murray and his wife Kim are all smiles after tying the knot at Dunblane Cathedral. Kim shows her delight after marrying British tennis star Murray on Saturday afternoon. The happy couple are covered in confetti as they make their way out of the wedding ceremony. Murray waves to his fans after making his way out of the cathedral following his wedding. Hundreds of well-wishers gathered on the streets around the cathedral and one of the biggest cheer so far has been reserved for Judy Murray, who arrived just before 4.00pm looking resplendent in a white, to-the-knee overcoat with a detailed gold dress.The outfit was topped off with a dramatic gold hat. This morning, the historic cathedral saw florists toting huge bouquets of flowers arrive to decorate the church - as hailstones and showers battered the venue. Clearly unable to contain his excitement this morning, tennis ace Murray posted a tweet to his 2.98 million followers which shows his plans for the day - through the use of emojis. He tweeted an umbrella, a picture of a church, a ring, a kiss, cake, drinks including beer, cocktails and wine and ends with hearts, a face blowing a kiss and several Zzzz icons for sleep."""

