# Computer Graphics (Számítógépes Grafika)
Solutions for Computer Graphics subject's 3 homework tasks during my second year of Computer Engineering BSc at BME (Spring 2020)

The specifications can be found in Hungarian, under the English ones.

## Running instructions
Copy the desired solution's ```.cpp``` file to the ```GraphicsHomeworkPackage/Programs/Skeleton/Skeleton``` folder and rename it to ```Skeleton.cpp```, then run ```Skeleton.sln``` which can be found in the ```GraphicsHomeworkPackage/Programs/Skeleton``` folder.

Please mind that [Visual Studio](https://visualstudio.microsoft.com/vs/) is required for the program to work.

- - - -

## Specifications:
### Homework 1 - Sirius triangle drawing
Aliens from the Sirius star constellation love the 2D drawing package of Word, Powerpoint, etc., but they cannot use it directly since their geometry differs from the Euclidean geometry popular in planet Earth. You are contracted to make the necessary adaptation for the simple case when a triangle defined by three points is to be drawn. Fortunately, they also provided a dictionary to pair the Sirius geometric terms to our familiar Euclidean geometric terms: 
The Sirius plane is a unit radius circle of the Euclidean plane, which is called the base circle.
Points of the Euclidean plane can be identified by a complex number. Points of the Sirius plane are identified by a complex number of absolute value less than 1. The metric of the Euclidean plane is|dz|=sqrt(dx^2+dy^2), while the metric of the Sirius plane is |dz|/(1-|z|^2) = sqrt(dx^2+dy^2)/(1 - x^2-y^2)
The Sirius line is an circular arc in Euclidean geometry, which arrives at the base circle perpendicularly. 
The program should render a filled Sirius triangle defined by three mouse clicks. The boundary of the Sirius triangle should be drawn with a different color. The lengths of the edges of the triangle should be printed on the standard output.

My solution scored 1 point out of 2.

Official solution [video](https://www.youtube.com/watch?v=_cCr-LF9QU0) of the first homework.

### Homework 2 - Light tube simulator
My solution scored 3 points out of 3.

Official Solution [video](https://www.youtube.com/watch?v=hTCNJaP_QNk) of the second homework.

### Homework 3 - Virus killing antibody
My solution scored 2 points out of 4.

Official solution [video](https://www.youtube.com/watch?v=NckDUAAUu4o) of the third homework.

- - - -

## Specifications in Hungarian:
### Első feladat - Sziriusz háromszög rajzoló
A Szíriusz csillagképből érkező földönkívülieknek megtetszett a Word, Powerpoint, stb. 2D-s rajzoló programja, de azt közvetlenül nem használhatják, ugyanis ők más geometriával dolgoznak. Önt bízták meg a program adaptálásával arra az egyszerű esetre, amikor három pontra egy háromszöget kell illeszteni. Szerencsére adtak egy szótárat, ami a szíriuszi geometriai fogalmakat megfelelteti az általunk használt euklideszi geometriai fogalmaknak:

A Szíriusz sík az euklideszi sík egység sugarú köre, amit alapkörnek neveznek.

Miközben az euklideszi síkot a komplex számokkal lehet számszerűsíteni, a Szíriusz síkot az egységnél kisebb abszolút értékű komplex számokkal. Amíg az euklideszi sík metrikája |dz|=sqrt(dx^2+dy^2) a Szíriusz síké |dz|/(1-|z|^2).

A Szíriusz egyenes egy olyan euklideszi körív, amely az alapkörre merőlegesen érkezik.

A feladat három egér klikk után a három pont által definiált háromszöget a háttértől eltérő színnel kitölteni, az éleket ezektől eltérő színnel felrajzolni, és a szabványos kimenetre a három szögeit és oldalainak hosszát kiírni.

Beadási határidő: 2020. 04. 04. 23:59

A megoldásom 1 pontot ért a 2-ből.

Az első házi hivatalos megoldása: [video](https://www.youtube.com/watch?v=_cCr-LF9QU0)

### Második feladat - Napfénycső szimulátor
A Velux a napfénycsöveit a beépítés előtt virtuális világban kívánja tesztelni. A feladatot önre osztotta a sors arra az esetre, amikor a napfénycső ezüst egyköpenyű hiperboloid és a nap közvetlenül nem világít be a szobába. A napfénycső által megvilágított szobában legalább három objektum van, amelyek szabadon megválaszthatók, de nem lehetnek gömbök. Az objektumokból legalább az egyik optikailag sima arany. A rücskös anyagok a diffúz+PhongBlinn modellt követik. Az nap+égbolt fénye konstans égbolt radianciával és a nap irányának környékén erősödő nap radianciával jellemezhető. Az égbolt fény csak a csövön juthat be a szobába, ahol csak egy ambiens fényforrás van jelen. Feladat, a szobában lévő virtuális kamerából lefényképezni a látványt egy olyan pillanatban, amikor a nap magasan jár és ezért jut elég fény a szobába.

Ezüst: n/k: 0.14/4.1, 0.16/2.3, 0.13/3.1

Arany: n/k: 0.17/3.1, 0.35/2.7, 1.5/1.9

Pontozás:

Szoba rücskös és optikailag sima anyagú tárgyakkal: 1p

Napfénycső és égbolt megvilágítás szimulációja: 2p

Beadási határidő: 2020. 04. 21. 23:59

A megoldásom 3 pontot ért a 3-ból.

A második házi hivatalos megoldása: [video](https://www.youtube.com/watch?v=hTCNJaP_QNk)

### Harmadik feladat - Vírusölő antitest
Készítsen „antitest vírust öl” játékot, amely egy textúrázott gömb vagy henger belsejében zajlik, amit pont fényforrások világítanak meg. A vírus teste haragosan hullámzó gömb, a nyúlványok Tractricoid alakzatok, amelyek a hullámzó felületre mindig merőlegesek. A nyúlványok egyenletesen fedik be a felületet. A gömb és a nyúlványok textúrázottak diffúz/spekuláris típusúak. A vírus a saját tengelye körül forog állandó szögsebességgel, ezen kívül a testén kívüli pivot pont körül is, amit a [cos(t), sin(t/2), sin(t/3), sin(t/5)] (vigyázat nem normalizált!) kvaternió ad meg (t az idő sec-ben). Az antitest tetraéder Helge von Koch típusú felosztással két szinten. Az antitest tüskéinek nyújtogatásával kelt félelmet. Az antitest saját tengelye körül forog és Brown mozgással halad előre, az x, X, y, Y, z, Z billentyűket lenyomva tartva a haladás az adott (kisbetű: pozitív, nagybetű: negatív) irányba valószínűbb. A Brown mozgás sebességvektora véletlenszerű és 0.1 sec-ként változik. Ha az antitest alaptetraéderének befoglaló gömbje és az alapállatú vírus gömbje ütközik, a vírus elpusztul, azaz mozgása megszűnik.

Pontozás:

Vírustest+korona: 1 pont

Vírustest+korona animáció: 1 pont

Antitest: 1 pont

Antitest animáció: 1 pont

Beadási határidő: 2020. 05. 15. 23:59

A megoldásom 2 pontot ért a 4-ből.

A harmadik házi hivatalos megoldása: [video](https://www.youtube.com/watch?v=NckDUAAUu4o)
