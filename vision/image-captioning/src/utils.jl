function load_data(base_path::String,num_samples=10000,punctuation=['.'])
    image_path = string(base_path,"train2014/")
    caption_path = string(base_path,"annotations/captions_train2014.json")
    
    json_data = JSON.parsefile(caption_path)
    captions = json_data["annotations"][1:num_samples]
    images = json_data["images"]
    
    data = [] #(caption,image) tuple
    count = 0
    for caption_data in captions
	count += 1
	println("Count : $(count)")
        image = [img for img in images if img["id"] == caption_data["image_id"]][1]
        caption = caption_data["caption"]
        for p in punctuation
        	caption = replace(caption,"$p"=>"")
        end
        caption = string("<s> ",caption," </s>")
        img_path = string(image_path,image["file_name"])
                
        push!(data,(caption,img_path))
    end
    
    return data
end

function get_mb_captions(idx)
    cap = tokenized_captions[idx]
    mb_captions = []
    # Convert to - Array[SEQ_LEN] with each element - [V,BATCH_SIZE]
    for i in 1:SEQ_LEN
        # Extract and form a batch of each word in sequence
        words = hcat([onehotword(sentence[i]) for sentence in cap]...)
        push!(mb_captions,words)
    end
    
    mb_captions
end
