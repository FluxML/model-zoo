using Cascadia, Gumbo, HTTP

pages = Dict(
  :en => ["Wikipedia", "Osama_bin_Laden_(elephant)", "List_of_lists_of_lists", "Josephine_Butler", "Canadian_football", "Judaism"],
  :it => ["Wikipedia", "Ludovico_Einaudi", "Filosofia_della_scienza", "Pizza", "Effie_Gray", "Galeazzo_Maria_Sforza", "Ebraismo"],
  :fr => ["Wikipedia", "Philosophie_des_sciences", "Seconde_Guerre_mondiale", "Eric_Hakonsson"],
  :es => ["Wikipedia", "Chorizo", "Historia_de_Barcelona", "Espania", "Las_Vegas_Strip", "Judaismo"],
  :da => ["Wikipedia", "H.C._Andersen", "L.A._Ring", "Jiangxi", "NATO", "Thomas_Edison", "Bangladesh"])

rawpage(url) = parsehtml(String(HTTP.get(url).body)).root

content(url) = join((collect(nodeText(m) for m in eachmatch(sel".mw-parser-output > p", rawpage(url)))), "\n")

cd(@__DIR__)
mkpath("corpus")

for (lang, ps) in pages
    open("corpus/$lang.txt", "w") do io
        for p in ps
            write(io, content("https://$lang.wikipedia.org/wiki/$p"))
        end
    end
end
