from transformers import AutoModelForCausalLM as Model

# pretrained = "distilbert/distilgpt2"

# model = Model.from_pretrained(pretrained) #load from remote - works fine

# model.save_pretrained("local_model",safe_serialization=False) #works fine

# del model

model  = Model.from_pretrained("local_model")

# model.save_pretrained("local_model2",safe_serialization=False) #raises SafeTensorError

model.save_pretrained("local_model",safe_serialization=False) #raises SafeTensorError
