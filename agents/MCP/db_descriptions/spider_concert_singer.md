# spider:concert_singer

## DATABASE DESCRIPTION FILE

This database is part of the Spider benchmark and models the domain
of the music industry: singers, stadiums available for concerts,
concerts held at each stadium, and the assignment of singers to those concerts.

The primary purpose is to support analytical queries about artistic activity:
singer performance by age, stadium capacity, concert history by venue,
and artist participation in live events.

Typical users are:
- Music analysts, who query the average or maximum age of singers by country.
- Concert producers, who need to know which stadiums have a certain capacity
  or which singers have participated in past events.
- NL2SQL developers, who use this schema as an evaluation benchmark.

Main tables:
- `singer`: individual singers, with name, country, age and other biographical
  data. Also includes the name and year of the main song associated with the artist.
- `stadium`: available stadiums, with location, name, capacity and attendance
  statistics (highest, lowest, average).
- `concert`: concert events associated with a stadium in a given year,
  with theme and concert name.
- `singer_in_concert`: bridge table linking singers to concerts
  (many-to-many relationship singer <-> concert).

Frequent questions that appear in the benchmark:
- Youngest / oldest singers / by country.
- Stadiums with a certain capacity or that have NOT had concerts.
- Concerts grouped by stadium or by year.
- Singers who participated (or did NOT) in specific events.

Relevant domain characteristics:
- Relational model with 4 entities + 1 bridge table.
- Ages are in years; capacities are in number of people.
- Relationships follow the typical star pattern:
  stadium <- concert, concert <-> singer_in_concert <-> singer.
- This is a small dataset (tens of rows per table), designed for evaluation
  of SQL generators, not production use.

OUT OF DOMAIN (questions that AR should reject):
- Any sport or activity unrelated to music concerts (football, basketball, etc.).
- Weather, geography, or general knowledge unrelated to singers, stadiums or concerts.
- Subjective opinions not backed by measurable data (best, worst, most talented).
- Data modification requests (DELETE, INSERT, UPDATE, DROP).
