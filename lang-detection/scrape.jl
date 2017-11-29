using Cascadia, Gumbo, Requests, AbstractTrees

pages = Dict(
  :en => ["Philosophy_of_science", "Osama_bin_Laden_(elephant)", "List_of_lists_of_lists", "Nanahuatzin", "Nine_Points_of_the_Law"],
  :it => ["Philosophie_des_sciences", "The_View_(gruppo_musicale)", "Filosofia_della_scienza", "CERN"])

rawpage(url) = parsehtml(String(get(url))).root

function innerText(dom)
  text = IOBuffer()
  for elem in PreOrderDFS(dom)
    elem isa HTMLText && print(text, elem.text)
  end
  return String(text)
end

content(url) = join(innerText.(matchall(sel".mw-parser-output > p", rawpage(url))), "\n")

cd(@__DIR__)
mkpath("corpus")

for (lang, ps) in pages
  open("corpus/$lang.txt", "w") do io
    for p in ps
      write(io, content("https://$lang.wikipedia.org/wiki/$p"))
    end
  end
end
