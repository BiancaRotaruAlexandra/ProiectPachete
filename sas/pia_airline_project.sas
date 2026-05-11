/**********************************************************************
  Proiect Pachete Software – parte SAS
  Analiza activitatii aeriene (PIA 2026) / extindere retea

  INSTRUCTIUNI:
  1. Modifica %projroot astfel incat sa indice folderul unde se afla
     subfolderul data\ cu fisierul CSV (acelasi ca pentru Streamlit).
  2. Ruleaza intreg scriptul in SAS Studio / SAS OnDemand / SAS EG.

  Fisier sursa CSV (aceleasi coloane ca in aplicatia Python):
  PIA_2026_Advanced_Kaggle_Dataset.csv
**********************************************************************/

/* ---------- Cale proiect (EDITATI aici) ---------- */
%let projroot = C:\Users\Andreea\Desktop\proiect-pachete\ProiectPachete;
%let csvfile  = &projroot.\data\PIA_2026_Advanced_Kaggle_Dataset.csv;

/* ---------- 1. Creare set de date SAS din fisier extern ---------- */
PROC IMPORT DATAFILE="&csvfile"
    OUT=WORK.flights_raw
    DBMS=CSV REPLACE;
    GETNAMES=YES;
    GUESSINGROWS=MAX;
RUN;

TITLE 'Primele inregistrari (import CSV)';
PROC PRINT DATA=WORK.flights_raw(OBS=10);
RUN;
TITLE;

/* ---------- 2. Formate definite de utilizator ---------- */
PROC FORMAT;
    VALUE $delay_fmt
        'Severe'   = 'Intarziere severa'
        'Moderate' = 'Intarziere moderata'
        'Minor'    = 'Intarziere minora'
        'None'     = 'Fara intarziere'
        OTHER      = 'Necunoscut';

    VALUE delay_bin
        LOW-<30  = 'Sub 30 min'
        30-<120  = '30-119 min'
        120-HIGH = '120+ min';

    VALUE revenue_band
        LOW-<50000   = 'Venit mic'
        50000-<200000 = 'Venit mediu'
        200000-HIGH = 'Venit mare';

    VALUE $ontime_fmt
        'On Time' = 'La timp'
        'Delayed' = 'Intarziat'
        OTHER     = 'Alt status';
RUN;

/* ---------- 3–5. Procesare iterativa / conditionala, subset, functii SAS, array ---------- */
DATA WORK.flights_prep;
    SET WORK.flights_raw;

    /* Subset logic: pastram doar zboruri cu pasageri pozitivi */
    IF Passengers <= 0 THEN DELETE;

    /* Functii SAS: concatenare ruta, categorii */
    Route = CATX(' -> ', Departure_City, Arrival_City);
    Delay_Log = LOG(Delay_Minutes + 1);

    /* Calcul factor incarcare evitat coloana cu % din CSV – folosim Seat_Capacity */
    IF Seat_Capacity > 0 THEN Load_Ratio = Passengers / Seat_Capacity;

    /* Array + DO: indechi compozit din operatii element-wise (cerinta „masive”) */
    ARRAY ops {*} Delay_Minutes Revenue_USD Fuel_Consumption_Liters;
    Composite_Ops_Index = 0;
    DO i = 1 TO DIM(ops);
        IF NOT MISSING(ops{i}) THEN Composite_Ops_Index + SQRT(MAX(ops{i}, 0));
    END;
    DROP i;

    /* Flag binar intarziere (pentru PROC LOGISTIC) */
    IF On_Time_Status = 'Delayed' THEN Is_Delayed = 1;
    ELSE Is_Delayed = 0;

    FORMAT Delay_Category $delay_fmt.
           Revenue_USD revenue_band.
           On_Time_Status $ontime_fmt.;
RUN;

/* Subset explicit – zboruri internationale */
DATA WORK.intl_only;
    SET WORK.flights_prep;
    WHERE Route_Type = 'International';
RUN;

TITLE 'Subset: zboruri internationale';
PROC FREQ DATA=WORK.intl_only;
    TABLES Route_Type On_Time_Status / NOPERCENT NOROW NOCOL;
RUN;
TITLE;

/* ---------- 6. Combinare seturi – DATA MERGE + lookup ---------- */
PROC SUMMARY DATA=WORK.flights_prep NWAY;
    CLASS Aircraft_Type;
    VAR Revenue_USD Delay_Minutes Passengers;
    OUTPUT OUT=WORK.aircraft_profile
        MEAN(Revenue_USD)=Avg_Revenue
        MEAN(Delay_Minutes)=Avg_Delay
        MEAN(Passengers)=Avg_Pax
        SUM(Revenue_USD)=Sum_Revenue;
RUN;

PROC SORT DATA=WORK.flights_prep; BY Aircraft_Type; RUN;
PROC SORT DATA=WORK.aircraft_profile; BY Aircraft_Type; RUN;

DATA WORK.flights_merged;
    MERGE WORK.flights_prep (IN=a)
          WORK.aircraft_profile (IN=b);
    BY Aircraft_Type;
    IF a;
    Rev_vs_Fleet_Avg = Revenue_USD - Avg_Revenue;
RUN;

/* ---------- 7. PROC SQL – join si agregari ---------- */
PROC SQL;
    CREATE TABLE WORK.sql_route_summary AS
    SELECT
        Departure_City,
        Arrival_City,
        COUNT(*) AS Num_Flights,
        SUM(Revenue_USD) AS Total_Revenue,
        MEAN(Delay_Minutes) AS Avg_Delay,
        MEAN(Customer_Rating) AS Avg_Rating
    FROM WORK.flights_prep
    GROUP BY Departure_City, Arrival_City
    HAVING CALCULATED Num_Flights >= 3
    ORDER BY Total_Revenue DESC;

    CREATE TABLE WORK.sql_join_demo AS
    SELECT a.Flight_ID,
           a.Route,
           a.Revenue_USD,
           b.Avg_Revenue AS Fleet_Avg_Revenue_Type
    FROM WORK.flights_prep AS a
    INNER JOIN WORK.aircraft_profile AS b
        ON a.Aircraft_Type = b.Aircraft_Type;
QUIT;

TITLE 'Top rute dupa venit (PROC SQL)';
PROC PRINT DATA=WORK.sql_route_summary(OBS=15);
RUN;
TITLE;

/* ---------- 8. Macro – procesare iterativa pe liste de variabile ---------- */
%MACRO describe_numeric(vars);
    %LET list = &vars;
    PROC MEANS DATA=WORK.flights_prep N MEAN STD MIN MAX;
        VAR &list;
    RUN;
%MEND describe_numeric;

%describe_numeric(Delay_Minutes Revenue_USD Passengers Customer_Rating);

/* ---------- 9. Raportare ---------- */
TITLE 'Raport pe tip ruta si status punctualitate';
PROC REPORT DATA=WORK.flights_prep NOWINDOWS HEADLINE;
    COLUMN Route_Type On_Time_Status Revenue_USD Delay_Minutes Passengers;
    DEFINE Route_Type / GROUP 'Tip ruta';
    DEFINE On_Time_Status / GROUP 'Status';
    DEFINE Revenue_USD / SUM FORMAT=DOLLAR12. 'Venit total';
    DEFINE Delay_Minutes / MEAN FORMAT=8.1 'Intarziere med.';
    DEFINE Passengers / SUM 'Pasageri';
RUN;
TITLE;

PROC PRINT DATA=WORK.flights_merged(OBS=12);
    VAR Flight_ID Route Aircraft_Type Revenue_USD Avg_Revenue Rev_vs_Fleet_Avg;
RUN;

/* ---------- 10. Proceduri statistice ---------- */
PROC MEANS DATA=WORK.flights_prep NOPRINT NWAY;
    CLASS Route_Type Delay_Category;
    VAR Revenue_USD Delay_Minutes Customer_Rating;
    OUTPUT OUT=WORK.stats_by_route
        MEAN(Revenue_USD)=Mean_Revenue
        MEAN(Delay_Minutes)=Mean_Delay
        MEAN(Customer_Rating)=Mean_Rating
        N(Revenue_USD)=N_Revenue
        MIN(Revenue_USD)=Min_Revenue
        MAX(Revenue_USD)=Max_Revenue;
RUN;

PROC CORR DATA=WORK.flights_prep PEARSON;
    VAR Revenue_USD Delay_Minutes Passengers Ticket_Price_USD Customer_Rating;
    WITH Fuel_Consumption_Liters Flight_Duration_Minutes;
RUN;

TITLE 'Regresie multipla – venit vs predictor numerici';
PROC REG DATA=WORK.flights_prep;
    MODEL Revenue_USD = Passengers Ticket_Price_USD Flight_Duration_Minutes Delay_Minutes;
RUN;
TITLE;

TITLE 'Regresie logistica – probabilitate intarziere';
PROC LOGISTIC DATA=WORK.flights_prep DESC;
    CLASS Route_Type Aircraft_Type Weather_Condition / PARAM=REF REF=FIRST;
    MODEL Is_Delayed = Flight_Duration_Minutes Passengers
                       Route_Type Aircraft_Type Weather_Condition
                       / SELECTION=STEPWISE RSQUARE;
RUN;
TITLE;

/* ---------- 11. SAS /STAT – clusterizare (FASTCLUS) ---------- */
PROC FASTCLUS DATA=WORK.flights_prep MAXCLUSTERS=4 MAXITER=100 OUT=WORK.clusters_out;
    VAR Revenue_USD Passengers Delay_Minutes Fuel_Consumption_Liters Customer_Rating;
RUN;

TITLE 'Profil clustere – medii';
PROC MEANS DATA=WORK.clusters_out;
    CLASS CLUSTER;
    VAR Revenue_USD Passengers Delay_Minutes Customer_Rating;
RUN;
TITLE;

/* ---------- 12. Grafice – PROC SGPLOT ---------- */
ODS GRAPHICS ON;

TITLE 'Distributie venit';
PROC SGPLOT DATA=WORK.flights_prep;
    HISTOGRAM Revenue_USD / BINWIDTH=50000 FILLATTRS=(COLOR=CX4682B4);
    DENSITY Revenue_USD / TYPE=NORMAL;
RUN;
TITLE;

TITLE 'Venit total pe tip ruta';
PROC SGPLOT DATA=WORK.flights_prep;
    VBAR Route_Type / RESPONSE=Revenue_USD STAT=SUM FILLATTRS=(COLOR=CX6BA292);
RUN;
TITLE;

TITLE 'Pasageri vs venit (culoare = punctualitate)';
PROC SGPLOT DATA=WORK.flights_prep;
    SCATTER X=Passengers Y=Revenue_USD / GROUP=On_Time_Status MARKERATTRS=(SIZE=3);
RUN;
TITLE;

ODS GRAPHICS OFF;

/* ---------- 13. SQL – indicator outer-style (FULL join demo pe subset mic) ---------- */
DATA WORK.lookup_city;
    LENGTH City $40 Role $10;
    City='Dubai'; Role='Hub'; OUTPUT;
    City='London'; Role='Hub'; OUTPUT;
    City='Paris'; Role='Regional'; OUTPUT;
RUN;

PROC SQL;
    CREATE TABLE WORK.city_role AS
    SELECT COALESCE(p.Departure_City, l.City) AS City,
           l.Role,
           COUNT(*) AS Departures
    FROM WORK.flights_prep AS p
    FULL JOIN WORK.lookup_city AS l
        ON p.Departure_City = l.City
    GROUP BY COALESCE(p.Departure_City, l.City), l.Role;
QUIT;

TITLE 'Demonstratie FULL JOIN pe hub-uri';
PROC PRINT DATA=WORK.city_role(OBS=20);
RUN;
TITLE;

/**********************************************************************
  Mapare rapida cerinte minime SAS (nota 5+):

  1) Set SAS din fisier extern          -> PROC IMPORT
  2) Formate utilizator                 -> PROC FORMAT ($delay_fmt, revenue_band...)
  3) Iterativ / conditional             -> DATA step DO loops, IF/WHERE
  4) Subseturi                          -> DATA intl_only; WHERE ...
  5) Functii SAS                        -> CATX, LOG, MEAN in SQL/PROC
  6) Combinare merge                    -> MERGE BY Aircraft_Type
  7) SQL                                -> CREATE TABLE ... JOIN / GROUP BY
  8) Array                              -> ARRAY nums {*} ... DO i=1 TO DIM
  9) Raportare                          -> PROC REPORT, PROC PRINT
 10) Statistic                          -> PROC MEANS, CORR, REG, LOGISTIC
 11) Grafice                            -> PROC SGPLOT
 12) ML / cluster                       -> PROC FASTCLUS

  Adaptati path-ul %projroot si interpretati economic rezultatele in Word/PDF.
**********************************************************************/
