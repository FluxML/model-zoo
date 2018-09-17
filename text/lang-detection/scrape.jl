using Cascadia, Gumbo, HTTP, AbstractTrees

pages = Dict(
  :en => ["Wikipedia", "Osama_bin_Laden_(elephant)", "List_of_lists_of_lists", "Josephine_Butler", "Canadian_football", "Judaism"],
  :it => ["Wikipedia", "Ludovico_Einaudi", "Filosofia_della_scienza", "Pizza", "Effie_Gray", "Galeazzo_Maria_Sforza", "Ebraismo"],
  :fr => ["Wikipedia", "Philosophie_des_sciences", "Seconde_Guerre_mondiale", "Eric_Hakonsson"],
  :es => ["Wikipedia", "Chorizo", "Historia_de_Barcelona", "Espana", "Las_Vegas_Strip", "Judaismo"],
  :da => ["Wikipedia", "H.C._Andersen", "L.A._Ring", "Jiangxi", "NATO", "Thomas_Edison", "Bangladesh"])

rawpage(url) = parsehtml(String(HTTP.get(url).body)).root

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
