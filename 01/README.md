# Dokumentation

## README
Python Version 3.10.2
### Packages
- numpy=1.22.3
- cv2=4.5.5
- math=3.10.2
- random=3.10.2

## Code-Erklärung
### Programmablauf
* Erzeugen eines Bildes mit Grauwertverlauf
* Definition Geschwindigkeitsfaktor und initiale Richtung, sowie Startpunkt für die Bewegung
* Festlegung Kantenlänge Quadrat und Füllung aus Mitte Grauwertbild übernehmen
* Zeichnen der beiden ortsfesten Quadrate (Funktion ***rect(...)***)
* Erzeugen eines neuen Fensters, automatischer Größe, mit Name: "An interesting title"
* Schleife mit Abbruch auf Taste "q"
    * Original Grauwertbild kopieren
    * Neue Position des bewegten Quadrates ermitteln
    * Bild im Fenster anzeigen

### Grauwertverlaufsbild erzeugen


* Anlegen eines zweidimensionalen Bildarrays ***gradient*** mit ***height*** Zeilen und ***width*** Spalten
* Gleichmäßiges füllen der Spalten mit aufsteigendem Grauwert von links nach rechts (links schwarz, rechts weiß)
* Da der maximale Grauwert 255 ist, wird die Breite des Bildes normalisiert, sodass bei jeder Breite der Gradient gleichmäßig ist.

``` python
for x in range(width):
    gradient[:, x] = int((x/width) * 255)
```

### Geschwindigkeit, Richtung und Startpunkt


* Konstanter Geschwindigkeitsfaktor ***speedFaktor*** auf 10, d.h. 10 Pixel/Frame
* Kantenlänge der Quadrate ***squareW*** auf 50
* Füllung ***square*** entsprechend Bildmitte Grauwertbild
* Aktuelle Position ***xind***, ***yind*** des bewegten Quadrates initial auf Bildmitte
* Winkel ***direction*** über Zufallsfunktion ***random*** multipliziert mit 2Pi berechnen
* Die Richtungen ergebn sich zu:
    * X-Richtung ***speedX*** = cos(direction)
    * Y-Richtung ***speedY*** entsprechend sin(direction)
    * siehe [hier](https://de.wikipedia.org/wiki/Polarkoordinaten#Umrechnung_zwischen_Polarkoordinaten_und_kartesischen_Koordinaten)

``` python
speedFactor = 10
direction = random.random() * 2 * math.pi
(speedX, speedY) = (math.cos(direction) * speedFactor, math.sin(direction) * speedFactor)

squareW = 50
square = gradient[height//2:height//2+squareW, width//2:width//2+squareW]
yind = height//2-squareW//2
xind = width//2-squareW//2
```


### Beschreibung Funktion ***rect(...)***
* Ersetzen der Pixel im Grauwertbild an der Position ***pos*** für die Ausdehnung ***size*** durch ***cutout***. Pos und size bestehen jeweils aus einem Tupel mit X- und Y-Koordinate, ***size*** beschreibt die Ausdehnung von ***cutout***.

* Funktionsdefinition

``` python
def rect(pos, size, cutout):
    gradient[pos[0]:pos[0] + size[0], pos[1]:pos[1] + size[1]] = cutout
```

* Funktionsaufruf:

``` python
rect((50, 50), (squareW, squareW), square)
```

Ersetzen der Pixel ab X = 50, Y = 50 für die Ausdehnung in ***X = squareW***, ***Y = squareW*** mit square(Ausschnitt Mitte Originalbild)

### Beschreibung Funktion ***calSpeed(...)***
* Betrachtung ob das bewegte Quadrat den Bildbereich in X- oder Y-Richtung verlässt
* Am Rand wird gegebenenfalls die entsprechende Richtung (speedX, speedY) umgedreht (Abprallefekt)

``` python
def calSpeed(xind, yind, speedX, speedY):
    if xind+squareW >= width-speedFactor or xind-speedFactor < 0:
        speedX *= -1
    if yind+squareW >= height-speedFactor or yind-speedFactor < 0:
        speedY *= -1
    return (xind + speedX, yind + speedY, speedX, speedY)
```


* Eingangsparameter

 Parameter | Beschreibung 
 --------- | ------------ 
 xind | aktuelle X-Position 
 yind | aktuelle Y-Position 
 speedX | aktuelle X-Geschwindigkeit
 speedY | aktuelle Y-Geschwindigkeit

* Rückgagewerte

 Parameter | Beschreibung 
 --------- | ------------ 
 xind | neue X-Position 
 yind | neue Y-Position 
 speedX | neue X-Geschwindigkeit
 speedY | neue Y-Geschwindigkeit


### Schleife zur Bildanzeige

* Originalbild Array ***gradient*** auf ***result*** kopieren mittels `Array.copy()` Funktion, da diese deutlich performanter ist als andere Methoden des kopierens, siehe [hier](https://www.youtube.com/watch?v=Qgevy75co8c)
* Neue Position und Richtung durch Aufruf von ***calSpeed*** ermitteln
* Von dieser Position ausgehend Bildbereich in ***result*** durch ausgeschnittenen Teil ***square*** ersetzen und anzeigen

``` python
while True:
    result = gradient.copy()
    (xind, yind, speedX, speedY) = calSpeed(xind, yind, speedX, speedY)
    result[int(yind):int(yind+squareW), int(xind):int(xind+squareW)] = square
    
    cv2.imshow('An interesting title', result)

    if cv2.waitKey(10) == ord('q'):
        cv2.destroyAllWindows()
        break
```


* Beenden wenn Taste "q" gedrückt
