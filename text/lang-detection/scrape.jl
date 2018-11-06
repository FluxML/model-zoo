using Cascadia # For 'eachmatch' on HTMLElements and 'Selector' 
using Gumbo    # For 'parsehtml' and 'PreOrderDFS'
using HTTP     # For 'get'
using AbstractTrees

pages = Dict(
  :en => ["Wikipedia", "Osama_bin_Laden_(elephant)", "List_of_lists_of_lists", "Josephine_Butler", "Canadian_football", "Judaism"],
  :it => ["Wikipedia", "Ludovico_Einaudi", "Filosofia_della_scienza", "Pizza", "Effie_Gray", "Galeazzo_Maria_Sforza", "Ebraismo"],
  :fr => ["Wikipedia", "Philosophie_des_sciences", "Seconde_Guerre_mondiale", "Eric_Hakonsson"],
  :es => ["Wikipedia", "Chorizo", "Historia_de_Barcelona", "Espana", "Las_Vegas_Strip", "Judaismo"],
  :da => ["Wikipedia", "H.C._Andersen", "L.A._Ring", "Jiangxi", "NATO", "Thomas_Edison", "Bangladesh"])

function innerText(dom)
  text = ""
  for elem in PreOrderDFS(dom)
        if elem isa HTMLText
            text = string(text, elem.text)
        end
  end
  return text
end

rawpage(url) = parsehtml(String(HTTP.get(url).body)).root
content(url) = join(innerText.(eachmatch(Selector(".mw-parser-output > p"), rawpage(url) )))

cd(@__DIR__)
mkpath("corpus")

for (lang, ps) in pages
  open("corpus/$lang.txt", "w") do io
    for p in ps
      write(io, content("https://$lang.wikipedia.org/wiki/$p"))
    end
  end
end