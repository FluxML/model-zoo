using Cascadia, Gumbo, Requests, AbstractTrees

pages = Dict(
  :en => ["Wikipedia", "Osama_bin_Laden_(elephant)", "List_of_lists_of_lists", "Nine_Points_of_the_Law"],
  :it => ["Wikipedia", "Ludovico_Einaudi", "Filosofia_della_scienza", "Pizza"],
  :fr => ["Wikipedia", "Philosophie_des_sciences"],
  :es => ["Wikipedia", "Chorizo"],
  :da => ["Wikipedia", "H.C._Andersen"])

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
