CREATE TABLE raw_images (
  "id" SERIAL PRIMARY KEY,
  "experiment" varchar,
  "run" int,
  "epoch" int,
  "step" int,
  "batch" int,
  "uniqueId" varchar,
  "image" bytea,
  "timestamp" timestamp,
  "avgProbability" float DEFAULT 0,
  "avgClarity" float DEFAULT 0,
  "annotationScore" float DEFAULT 0,
  "fileName" varchar,
  "totalAnnotations" int DEFAULT 0
);


CREATE TABLE annotated_images (
  "id" SERIAL PRIMARY KEY,
  "rawImageId" int,
  "uniqueId" varchar,
  "label" varchar,
  "probability" float,
  "clarity" float,
  "userEmail" varchar ,
  "timestamp" timestamp
);

CREATE TABLE filter_options
(
	option varchar not null
);

create unique index filteroptions_option_uindex
	on "filter_options" (option);

ALTER TABLE annotated_images ADD FOREIGN KEY ("rawImageId") REFERENCES raw_images ("id");

ALTER TABLE annotated_images ADD FOREIGN KEY ("uniqueId") REFERENCES raw_images ("uniqueId");

ALTER TABLE annotated_images ADD FOREIGN KEY ("userEmail") REFERENCES "user" ("email");

ALTER TABLE "user" ADD "totalAnnotations" int default 0;

