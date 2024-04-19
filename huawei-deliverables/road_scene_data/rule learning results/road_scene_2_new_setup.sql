-- MySQL dump 10.13  Distrib 8.2.0, for macos13 (x86_64)
--
-- Host: database-1.cxcqxpvbnnwo.us-east-2.rds.amazonaws.com    Database: road_scene_2_new_setup
-- ------------------------------------------------------
-- Server version	5.5.5-10.6.14-MariaDB-log

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `1Nodes`
--

DROP TABLE IF EXISTS `1Nodes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `1Nodes` (
  `1nid` varchar(166) NOT NULL,
  `COLUMN_NAME` varchar(64) NOT NULL,
  `pvid` varchar(100) NOT NULL,
  `main` int(1) DEFAULT NULL,
  PRIMARY KEY (`1nid`),
  UNIQUE KEY `pvid` (`pvid`,`COLUMN_NAME`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `1Nodes`
--

LOCK TABLES `1Nodes` WRITE;
/*!40000 ALTER TABLE `1Nodes` DISABLE KEYS */;
/*!40000 ALTER TABLE `1Nodes` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `2Nodes`
--

DROP TABLE IF EXISTS `2Nodes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `2Nodes` (
  `2nid` varchar(267) DEFAULT NULL,
  `COLUMN_NAME` varchar(64) NOT NULL,
  `pvid1` varchar(100) NOT NULL DEFAULT '',
  `pvid2` varchar(100) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `main` int(11) DEFAULT NULL,
  PRIMARY KEY (`COLUMN_NAME`,`pvid1`,`pvid2`),
  KEY `index` (`pvid1`,`pvid2`,`TABLE_NAME`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `2Nodes`
--

LOCK TABLES `2Nodes` WRITE;
/*!40000 ALTER TABLE `2Nodes` DISABLE KEYS */;
INSERT INTO `2Nodes` VALUES ('Lane(frames0,cars0)','Lane','frames0','cars0','car_in_frame',1),('Lane(frames1,cars0)','Lane','frames1','cars0','car_in_frame',0),('near_level(frames0,cars0)','near_level','frames0','cars0','car_in_frame',1),('near_level(frames1,cars0)','near_level','frames1','cars0','car_in_frame',0),('Speed(frames0,cars0)','Speed','frames0','cars0','car_in_frame',1),('Speed(frames0,ego_cars0)','Speed','frames0','ego_cars0','ego_frame',1),('Speed(frames1,cars0)','Speed','frames1','cars0','car_in_frame',0),('Speed(frames1,ego_cars0)','Speed','frames1','ego_cars0','ego_frame',0),('speed_diff(frames0,cars0)','speed_diff','frames0','cars0','car_in_frame',1),('speed_diff(frames1,cars0)','speed_diff','frames1','cars0','car_in_frame',0);
/*!40000 ALTER TABLE `2Nodes` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `AttributeColumns`
--

DROP TABLE IF EXISTS `AttributeColumns`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `AttributeColumns` (
  `TABLE_NAME` varchar(64) NOT NULL,
  `COLUMN_NAME` varchar(64) NOT NULL,
  PRIMARY KEY (`TABLE_NAME`,`COLUMN_NAME`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `AttributeColumns`
--

LOCK TABLES `AttributeColumns` WRITE;
/*!40000 ALTER TABLE `AttributeColumns` DISABLE KEYS */;
INSERT INTO `AttributeColumns` VALUES ('car_in_frame','Lane'),('car_in_frame','near_level'),('car_in_frame','Speed'),('car_in_frame','speed_diff'),('ego_frame','Speed');
/*!40000 ALTER TABLE `AttributeColumns` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Attribute_Value`
--

DROP TABLE IF EXISTS `Attribute_Value`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Attribute_Value` (
  `COLUMN_NAME` varchar(269) DEFAULT NULL,
  `VALUE` varchar(30) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Attribute_Value`
--

LOCK TABLES `Attribute_Value` WRITE;
/*!40000 ALTER TABLE `Attribute_Value` DISABLE KEYS */;
INSERT INTO `Attribute_Value` VALUES ('Lane','lane_right'),('Lane','lane_middle'),('Lane','lane_left'),('near_level','1'),('near_level','3'),('near_level','4'),('near_level','2'),('near_level','5'),('Speed','high'),('Speed','medium'),('Speed','low'),('speed_diff','medium'),('speed_diff','high'),('speed_diff','low'),('Speed','high'),('Speed','medium'),('Speed','low'),('car_in_frame(frames0,cars0)','T'),('car_in_frame(frames0,cars0)','F'),('car_in_frame(frames1,cars0)','T'),('car_in_frame(frames1,cars0)','F'),('ego_frame(frames0,ego_cars0)','T'),('ego_frame(frames0,ego_cars0)','F'),('ego_frame(frames1,ego_cars0)','T'),('ego_frame(frames1,ego_cars0)','F'),('succ_frame(frames0,frames1)','T'),('succ_frame(frames0,frames1)','F');
/*!40000 ALTER TABLE `Attribute_Value` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `EntityTables`
--

DROP TABLE IF EXISTS `EntityTables`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `EntityTables` (
  `TABLE_NAME` varchar(64) NOT NULL,
  `COLUMN_NAME` varchar(64) NOT NULL,
  PRIMARY KEY (`TABLE_NAME`,`COLUMN_NAME`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `EntityTables`
--

LOCK TABLES `EntityTables` WRITE;
/*!40000 ALTER TABLE `EntityTables` DISABLE KEYS */;
INSERT INTO `EntityTables` VALUES ('cars','car_id'),('ego_cars','ego_id'),('frames','f_id');
/*!40000 ALTER TABLE `EntityTables` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Expansions`
--

DROP TABLE IF EXISTS `Expansions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Expansions` (
  `pvid` varchar(40) NOT NULL,
  PRIMARY KEY (`pvid`),
  CONSTRAINT `Expansions_ibfk_1` FOREIGN KEY (`pvid`) REFERENCES `PVariables` (`pvid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Expansions`
--

LOCK TABLES `Expansions` WRITE;
/*!40000 ALTER TABLE `Expansions` DISABLE KEYS */;
/*!40000 ALTER TABLE `Expansions` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `FNodes`
--

DROP TABLE IF EXISTS `FNodes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `FNodes` (
  `Fid` varchar(199) NOT NULL,
  `FunctorName` varchar(64) DEFAULT NULL,
  `Type` varchar(5) DEFAULT NULL,
  `main` int(11) DEFAULT NULL,
  PRIMARY KEY (`Fid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `FNodes`
--

LOCK TABLES `FNodes` WRITE;
/*!40000 ALTER TABLE `FNodes` DISABLE KEYS */;
INSERT INTO `FNodes` VALUES ('car_in_frame(frames0,cars0)','car_in_frame','Rnode',1),('car_in_frame(frames1,cars0)','car_in_frame','Rnode',0),('ego_frame(frames0,ego_cars0)','ego_frame','Rnode',1),('ego_frame(frames1,ego_cars0)','ego_frame','Rnode',0),('Lane(frames0,cars0)','Lane','2Node',1),('Lane(frames1,cars0)','Lane','2Node',0),('near_level(frames0,cars0)','near_level','2Node',1),('near_level(frames1,cars0)','near_level','2Node',0),('Speed(frames0,cars0)','Speed','2Node',1),('Speed(frames0,ego_cars0)','Speed','2Node',1),('Speed(frames1,cars0)','Speed','2Node',0),('Speed(frames1,ego_cars0)','Speed','2Node',0),('speed_diff(frames0,cars0)','speed_diff','2Node',1),('speed_diff(frames1,cars0)','speed_diff','2Node',0),('succ_frame(frames0,frames1)','succ_frame','Rnode',1);
/*!40000 ALTER TABLE `FNodes` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `FNodes_pvars`
--

DROP TABLE IF EXISTS `FNodes_pvars`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `FNodes_pvars` (
  `Fid` varchar(267) DEFAULT NULL,
  `pvid` varchar(100) NOT NULL DEFAULT '',
  KEY `Fid` (`Fid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `FNodes_pvars`
--

LOCK TABLES `FNodes_pvars` WRITE;
/*!40000 ALTER TABLE `FNodes_pvars` DISABLE KEYS */;
INSERT INTO `FNodes_pvars` VALUES ('Lane(frames0,cars0)','frames0'),('Lane(frames1,cars0)','frames1'),('near_level(frames0,cars0)','frames0'),('near_level(frames1,cars0)','frames1'),('Speed(frames0,cars0)','frames0'),('Speed(frames0,ego_cars0)','frames0'),('Speed(frames1,cars0)','frames1'),('Speed(frames1,ego_cars0)','frames1'),('speed_diff(frames0,cars0)','frames0'),('speed_diff(frames1,cars0)','frames1'),('Lane(frames0,cars0)','cars0'),('Lane(frames1,cars0)','cars0'),('near_level(frames0,cars0)','cars0'),('near_level(frames1,cars0)','cars0'),('Speed(frames0,cars0)','cars0'),('Speed(frames0,ego_cars0)','ego_cars0'),('Speed(frames1,cars0)','cars0'),('Speed(frames1,ego_cars0)','ego_cars0'),('speed_diff(frames0,cars0)','cars0'),('speed_diff(frames1,cars0)','cars0'),('car_in_frame(frames0,cars0)','frames0'),('car_in_frame(frames1,cars0)','frames1'),('ego_frame(frames0,ego_cars0)','frames0'),('ego_frame(frames1,ego_cars0)','frames1'),('succ_frame(frames0,frames1)','frames0'),('car_in_frame(frames0,cars0)','cars0'),('car_in_frame(frames1,cars0)','cars0'),('ego_frame(frames0,ego_cars0)','ego_cars0'),('ego_frame(frames1,ego_cars0)','ego_cars0'),('succ_frame(frames0,frames1)','frames1');
/*!40000 ALTER TABLE `FNodes_pvars` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `ForeignKeyColumns`
--

DROP TABLE IF EXISTS `ForeignKeyColumns`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ForeignKeyColumns` (
  `TABLE_NAME` varchar(64) NOT NULL,
  `COLUMN_NAME` varchar(64) NOT NULL,
  `REFERENCED_TABLE_NAME` varchar(64) NOT NULL,
  `REFERENCED_COLUMN_NAME` varchar(64),
  `CONSTRAINT_NAME` varchar(64) NOT NULL,
  `ORDINAL_POSITION` bigint(21) unsigned NOT NULL,
  PRIMARY KEY (`TABLE_NAME`,`COLUMN_NAME`,`REFERENCED_TABLE_NAME`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ForeignKeyColumns`
--

LOCK TABLES `ForeignKeyColumns` WRITE;
/*!40000 ALTER TABLE `ForeignKeyColumns` DISABLE KEYS */;
INSERT INTO `ForeignKeyColumns` VALUES ('car_in_frame','car_id','cars','car_id','car_in_frame_ibfk_2',2),('car_in_frame','f_id','frames','f_id','car_in_frame_ibfk_1',1),('ego_frame','ego_id','ego_cars','ego_id','ego_frame_ibfk_2',2),('ego_frame','f_id','frames','f_id','ego_frame_ibfk_1',1),('succ_frame','f_id1','frames','f_id','succ_frame_ibfk_1',1),('succ_frame','f_id2','frames','f_id','succ_frame_ibfk_2',2);
/*!40000 ALTER TABLE `ForeignKeyColumns` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `ForeignKeys_pvars`
--

DROP TABLE IF EXISTS `ForeignKeys_pvars`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ForeignKeys_pvars` (
  `TABLE_NAME` varchar(64) NOT NULL,
  `REFERENCED_TABLE_NAME` varchar(64) NOT NULL,
  `COLUMN_NAME` varchar(64) NOT NULL,
  `pvid` varchar(100) NOT NULL,
  `index_number` char(1) DEFAULT NULL,
  `ARGUMENT_POSITION` bigint(21) unsigned NOT NULL,
  PRIMARY KEY (`TABLE_NAME`,`pvid`,`ARGUMENT_POSITION`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ForeignKeys_pvars`
--

LOCK TABLES `ForeignKeys_pvars` WRITE;
/*!40000 ALTER TABLE `ForeignKeys_pvars` DISABLE KEYS */;
INSERT INTO `ForeignKeys_pvars` VALUES ('car_in_frame','cars','car_id','cars0','0',2),('car_in_frame','frames','f_id','frames0','0',1),('car_in_frame','frames','f_id','frames1','1',1),('ego_frame','ego_cars','ego_id','ego_cars0','0',2),('ego_frame','frames','f_id','frames0','0',1),('ego_frame','frames','f_id','frames1','1',1),('succ_frame','frames','f_id1','frames0','0',1),('succ_frame','frames','f_id2','frames0','0',2),('succ_frame','frames','f_id1','frames1','1',1),('succ_frame','frames','f_id2','frames1','1',2);
/*!40000 ALTER TABLE `ForeignKeys_pvars` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `FunctorSet`
--

DROP TABLE IF EXISTS `FunctorSet`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `FunctorSet` (
  `Fid` varchar(199) NOT NULL,
  PRIMARY KEY (`Fid`)
) ENGINE=MEMORY DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `FunctorSet`
--

LOCK TABLES `FunctorSet` WRITE;
/*!40000 ALTER TABLE `FunctorSet` DISABLE KEYS */;
INSERT INTO `FunctorSet` VALUES ('car_in_frame(frames0,cars0)'),('car_in_frame(frames1,cars0)'),('ego_frame(frames0,ego_cars0)'),('ego_frame(frames1,ego_cars0)'),('Lane(frames0,cars0)'),('Lane(frames1,cars0)'),('near_level(frames0,cars0)'),('near_level(frames1,cars0)'),('Speed(frames0,cars0)'),('Speed(frames0,ego_cars0)'),('Speed(frames1,cars0)'),('Speed(frames1,ego_cars0)'),('speed_diff(frames0,cars0)'),('speed_diff(frames1,cars0)'),('succ_frame(frames0,frames1)');
/*!40000 ALTER TABLE `FunctorSet` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Groundings`
--

DROP TABLE IF EXISTS `Groundings`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Groundings` (
  `pvid` varchar(40) NOT NULL,
  `id` varchar(256) NOT NULL,
  PRIMARY KEY (`pvid`,`id`),
  CONSTRAINT `Groundings_ibfk_1` FOREIGN KEY (`pvid`) REFERENCES `PVariables` (`pvid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Groundings`
--

LOCK TABLES `Groundings` WRITE;
/*!40000 ALTER TABLE `Groundings` DISABLE KEYS */;
/*!40000 ALTER TABLE `Groundings` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `InputColumns`
--

DROP TABLE IF EXISTS `InputColumns`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `InputColumns` (
  `TABLE_NAME` varchar(64) NOT NULL,
  `COLUMN_NAME` varchar(64) NOT NULL,
  `REFERENCED_TABLE_NAME` varchar(64),
  `REFERENCED_COLUMN_NAME` varchar(64),
  `CONSTRAINT_NAME` varchar(64) NOT NULL,
  `ORDINAL_POSITION` bigint(21) unsigned NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `InputColumns`
--

LOCK TABLES `InputColumns` WRITE;
/*!40000 ALTER TABLE `InputColumns` DISABLE KEYS */;
INSERT INTO `InputColumns` VALUES ('cars','car_id',NULL,NULL,'PRIMARY',1),('car_in_frame','car_id',NULL,NULL,'PRIMARY',2),('car_in_frame','f_id',NULL,NULL,'PRIMARY',1),('ego_cars','ego_id',NULL,NULL,'PRIMARY',1),('ego_frame','ego_id',NULL,NULL,'PRIMARY',2),('ego_frame','f_id',NULL,NULL,'PRIMARY',1),('frames','f_id',NULL,NULL,'PRIMARY',1),('succ_frame','f_id2',NULL,NULL,'PRIMARY',2),('succ_frame','f_id1',NULL,NULL,'PRIMARY',1);
/*!40000 ALTER TABLE `InputColumns` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `KeyColumns`
--

DROP TABLE IF EXISTS `KeyColumns`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `KeyColumns` (
  `TABLE_NAME` varchar(64) NOT NULL,
  `COLUMN_NAME` varchar(64) NOT NULL,
  `REFERENCED_TABLE_NAME` varchar(64),
  `REFERENCED_COLUMN_NAME` varchar(64),
  `CONSTRAINT_NAME` varchar(64) NOT NULL,
  `ORDINAL_POSITION` bigint(21) unsigned NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `KeyColumns`
--

LOCK TABLES `KeyColumns` WRITE;
/*!40000 ALTER TABLE `KeyColumns` DISABLE KEYS */;
INSERT INTO `KeyColumns` VALUES ('cars','car_id',NULL,NULL,'PRIMARY',1),('car_in_frame','car_id',NULL,NULL,'PRIMARY',2),('car_in_frame','car_id','cars','car_id','car_in_frame_ibfk_2',2),('car_in_frame','f_id',NULL,NULL,'PRIMARY',1),('car_in_frame','f_id','frames','f_id','car_in_frame_ibfk_1',1),('ego_cars','ego_id',NULL,NULL,'PRIMARY',1),('ego_frame','ego_id',NULL,NULL,'PRIMARY',2),('ego_frame','ego_id','ego_cars','ego_id','ego_frame_ibfk_2',2),('ego_frame','f_id',NULL,NULL,'PRIMARY',1),('ego_frame','f_id','frames','f_id','ego_frame_ibfk_1',1),('frames','f_id',NULL,NULL,'PRIMARY',1),('succ_frame','f_id2',NULL,NULL,'PRIMARY',2),('succ_frame','f_id2','frames','f_id','succ_frame_ibfk_2',2),('succ_frame','f_id1',NULL,NULL,'PRIMARY',1),('succ_frame','f_id1','frames','f_id','succ_frame_ibfk_1',1);
/*!40000 ALTER TABLE `KeyColumns` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `LatticeRNodes`
--

DROP TABLE IF EXISTS `LatticeRNodes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `LatticeRNodes` (
  `orig_rnid` varchar(267) DEFAULT NULL,
  `short_rnid` varbinary(4) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `LatticeRNodes`
--

LOCK TABLES `LatticeRNodes` WRITE;
/*!40000 ALTER TABLE `LatticeRNodes` DISABLE KEYS */;
INSERT INTO `LatticeRNodes` VALUES ('car_in_frame(frames0,cars0)',_binary 'a'),('car_in_frame(frames1,cars0)',_binary 'b'),('ego_frame(frames0,ego_cars0)',_binary 'c'),('ego_frame(frames1,ego_cars0)',_binary 'd'),('succ_frame(frames0,frames1)',_binary 'e');
/*!40000 ALTER TABLE `LatticeRNodes` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Many_OneRelationships`
--

DROP TABLE IF EXISTS `Many_OneRelationships`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Many_OneRelationships` (
  `TABLE_NAME` varchar(64) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Many_OneRelationships`
--

LOCK TABLES `Many_OneRelationships` WRITE;
/*!40000 ALTER TABLE `Many_OneRelationships` DISABLE KEYS */;
/*!40000 ALTER TABLE `Many_OneRelationships` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `NoPKeys`
--

DROP TABLE IF EXISTS `NoPKeys`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `NoPKeys` (
  `TABLE_NAME` varchar(64) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `NoPKeys`
--

LOCK TABLES `NoPKeys` WRITE;
/*!40000 ALTER TABLE `NoPKeys` DISABLE KEYS */;
/*!40000 ALTER TABLE `NoPKeys` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `NumEntityColumns`
--

DROP TABLE IF EXISTS `NumEntityColumns`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `NumEntityColumns` (
  `TABLE_NAME` varchar(64) NOT NULL,
  `num` bigint(21) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `NumEntityColumns`
--

LOCK TABLES `NumEntityColumns` WRITE;
/*!40000 ALTER TABLE `NumEntityColumns` DISABLE KEYS */;
INSERT INTO `NumEntityColumns` VALUES ('cars',1),('car_in_frame',2),('ego_cars',1),('ego_frame',2),('frames',1),('succ_frame',2);
/*!40000 ALTER TABLE `NumEntityColumns` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `PVariables`
--

DROP TABLE IF EXISTS `PVariables`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `PVariables` (
  `pvid` varchar(100) NOT NULL,
  `TABLE_NAME` varchar(100) DEFAULT NULL,
  `ID_COLUMN_NAME` varchar(100) DEFAULT NULL,
  `index_number` char(1) DEFAULT NULL,
  PRIMARY KEY (`pvid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `PVariables`
--

LOCK TABLES `PVariables` WRITE;
/*!40000 ALTER TABLE `PVariables` DISABLE KEYS */;
INSERT INTO `PVariables` VALUES ('cars0','cars','car_id','0'),('ego_cars0','ego_cars','ego_id','0'),('frames0','frames','f_id','0'),('frames1','frames','f_id','1');
/*!40000 ALTER TABLE `PVariables` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `RNodes`
--

DROP TABLE IF EXISTS `RNodes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `RNodes` (
  `rnid` varchar(267) DEFAULT NULL,
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `pvid1` varchar(100) NOT NULL DEFAULT '',
  `pvid2` varchar(100) NOT NULL DEFAULT '',
  `COLUMN_NAME1` varchar(64) NOT NULL DEFAULT '',
  `COLUMN_NAME2` varchar(64) NOT NULL DEFAULT '',
  `main` int(11) DEFAULT NULL,
  PRIMARY KEY (`TABLE_NAME`,`pvid1`,`pvid2`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `RNodes`
--

LOCK TABLES `RNodes` WRITE;
/*!40000 ALTER TABLE `RNodes` DISABLE KEYS */;
INSERT INTO `RNodes` VALUES ('car_in_frame(frames0,cars0)','car_in_frame','frames0','cars0','f_id','car_id',1),('car_in_frame(frames1,cars0)','car_in_frame','frames1','cars0','f_id','car_id',0),('ego_frame(frames0,ego_cars0)','ego_frame','frames0','ego_cars0','f_id','ego_id',1),('ego_frame(frames1,ego_cars0)','ego_frame','frames1','ego_cars0','f_id','ego_id',0),('succ_frame(frames0,frames1)','succ_frame','frames0','frames1','f_id1','f_id2',1);
/*!40000 ALTER TABLE `RNodes` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `RNodes_2Nodes`
--

DROP TABLE IF EXISTS `RNodes_2Nodes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `RNodes_2Nodes` (
  `rnid` varchar(267) DEFAULT NULL,
  `2nid` varchar(267) DEFAULT NULL,
  `main` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `RNodes_2Nodes`
--

LOCK TABLES `RNodes_2Nodes` WRITE;
/*!40000 ALTER TABLE `RNodes_2Nodes` DISABLE KEYS */;
INSERT INTO `RNodes_2Nodes` VALUES ('car_in_frame(frames0,cars0)','Lane(frames0,cars0)',1),('car_in_frame(frames1,cars0)','Lane(frames1,cars0)',0),('car_in_frame(frames0,cars0)','near_level(frames0,cars0)',1),('car_in_frame(frames1,cars0)','near_level(frames1,cars0)',0),('car_in_frame(frames0,cars0)','Speed(frames0,cars0)',1),('ego_frame(frames0,ego_cars0)','Speed(frames0,ego_cars0)',1),('car_in_frame(frames1,cars0)','Speed(frames1,cars0)',0),('ego_frame(frames1,ego_cars0)','Speed(frames1,ego_cars0)',0),('car_in_frame(frames0,cars0)','speed_diff(frames0,cars0)',1),('car_in_frame(frames1,cars0)','speed_diff(frames1,cars0)',0);
/*!40000 ALTER TABLE `RNodes_2Nodes` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `RNodes_MM_NotSelf`
--

DROP TABLE IF EXISTS `RNodes_MM_NotSelf`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `RNodes_MM_NotSelf` (
  `rnid` varchar(267) DEFAULT NULL,
  `TABLE_NAME` varchar(64) NOT NULL,
  `pvid1` varchar(100) NOT NULL,
  `pvid2` varchar(100) NOT NULL,
  `COLUMN_NAME1` varchar(64) NOT NULL,
  `COLUMN_NAME2` varchar(64) NOT NULL,
  `main` int(1) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `RNodes_MM_NotSelf`
--

LOCK TABLES `RNodes_MM_NotSelf` WRITE;
/*!40000 ALTER TABLE `RNodes_MM_NotSelf` DISABLE KEYS */;
INSERT INTO `RNodes_MM_NotSelf` VALUES ('car_in_frame(frames0,cars0)','car_in_frame','frames0','cars0','f_id','car_id',1),('car_in_frame(frames1,cars0)','car_in_frame','frames1','cars0','f_id','car_id',0),('ego_frame(frames0,ego_cars0)','ego_frame','frames0','ego_cars0','f_id','ego_id',1),('ego_frame(frames1,ego_cars0)','ego_frame','frames1','ego_cars0','f_id','ego_id',0);
/*!40000 ALTER TABLE `RNodes_MM_NotSelf` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `RNodes_MM_Self`
--

DROP TABLE IF EXISTS `RNodes_MM_Self`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `RNodes_MM_Self` (
  `rnid` varchar(267) DEFAULT NULL,
  `TABLE_NAME` varchar(64) NOT NULL,
  `pvid1` varchar(100) NOT NULL,
  `pvid2` varchar(100) NOT NULL,
  `COLUMN_NAME1` varchar(64) NOT NULL,
  `COLUMN_NAME2` varchar(64) NOT NULL,
  `main` int(1) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `RNodes_MM_Self`
--

LOCK TABLES `RNodes_MM_Self` WRITE;
/*!40000 ALTER TABLE `RNodes_MM_Self` DISABLE KEYS */;
INSERT INTO `RNodes_MM_Self` VALUES ('succ_frame(frames0,frames1)','succ_frame','frames0','frames1','f_id1','f_id2',1);
/*!40000 ALTER TABLE `RNodes_MM_Self` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `RNodes_MO_NotSelf`
--

DROP TABLE IF EXISTS `RNodes_MO_NotSelf`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `RNodes_MO_NotSelf` (
  `rnid` varchar(267) DEFAULT NULL,
  `TABLE_NAME` varchar(64) NOT NULL,
  `pvid1` varchar(100) NOT NULL,
  `pvid2` varchar(100) NOT NULL,
  `COLUMN_NAME1` varchar(64) NOT NULL,
  `COLUMN_NAME2` varchar(64) NOT NULL,
  `main` int(1) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `RNodes_MO_NotSelf`
--

LOCK TABLES `RNodes_MO_NotSelf` WRITE;
/*!40000 ALTER TABLE `RNodes_MO_NotSelf` DISABLE KEYS */;
/*!40000 ALTER TABLE `RNodes_MO_NotSelf` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `RNodes_MO_Self`
--

DROP TABLE IF EXISTS `RNodes_MO_Self`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `RNodes_MO_Self` (
  `rnid` varchar(267) DEFAULT NULL,
  `TABLE_NAME` varchar(64) NOT NULL,
  `pvid1` varchar(100) NOT NULL,
  `pvid2` varchar(100) NOT NULL,
  `COLUMN_NAME1` varchar(64) NOT NULL,
  `COLUMN_NAME2` varchar(64) NOT NULL,
  `main` int(1) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `RNodes_MO_Self`
--

LOCK TABLES `RNodes_MO_Self` WRITE;
/*!40000 ALTER TABLE `RNodes_MO_Self` DISABLE KEYS */;
/*!40000 ALTER TABLE `RNodes_MO_Self` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `RNodes_pvars`
--

DROP TABLE IF EXISTS `RNodes_pvars`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `RNodes_pvars` (
  `rnid` varchar(267) DEFAULT NULL,
  `pvid` varchar(100) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(100) DEFAULT NULL,
  `COLUMN_NAME` varchar(64) NOT NULL DEFAULT '',
  `REFERENCED_COLUMN_NAME` varchar(64) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `RNodes_pvars`
--

LOCK TABLES `RNodes_pvars` WRITE;
/*!40000 ALTER TABLE `RNodes_pvars` DISABLE KEYS */;
INSERT INTO `RNodes_pvars` VALUES ('car_in_frame(frames0,cars0)','frames0','frames','f_id','f_id'),('ego_frame(frames0,ego_cars0)','frames0','frames','f_id','f_id'),('succ_frame(frames0,frames1)','frames0','frames','f_id1','f_id'),('car_in_frame(frames1,cars0)','frames1','frames','f_id','f_id'),('ego_frame(frames1,ego_cars0)','frames1','frames','f_id','f_id'),('car_in_frame(frames0,cars0)','cars0','cars','car_id','car_id'),('car_in_frame(frames1,cars0)','cars0','cars','car_id','car_id'),('ego_frame(frames0,ego_cars0)','ego_cars0','ego_cars','ego_id','ego_id'),('ego_frame(frames1,ego_cars0)','ego_cars0','ego_cars','ego_id','ego_id'),('succ_frame(frames0,frames1)','frames1','frames','f_id2','f_id');
/*!40000 ALTER TABLE `RNodes_pvars` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `RelationTables`
--

DROP TABLE IF EXISTS `RelationTables`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `RelationTables` (
  `TABLE_NAME` varchar(64) NOT NULL,
  `SelfRelationship` int(21) DEFAULT NULL,
  `Many_OneRelationship` int(21) DEFAULT NULL,
  PRIMARY KEY (`TABLE_NAME`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `RelationTables`
--

LOCK TABLES `RelationTables` WRITE;
/*!40000 ALTER TABLE `RelationTables` DISABLE KEYS */;
INSERT INTO `RelationTables` VALUES ('car_in_frame',0,0),('ego_frame',0,0),('succ_frame',1,0);
/*!40000 ALTER TABLE `RelationTables` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Schema_Key_Info`
--

DROP TABLE IF EXISTS `Schema_Key_Info`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Schema_Key_Info` (
  `TABLE_NAME` varchar(64) NOT NULL,
  `COLUMN_NAME` varchar(64) NOT NULL,
  `REFERENCED_TABLE_NAME` varchar(64),
  `REFERENCED_COLUMN_NAME` varchar(64),
  `CONSTRAINT_NAME` varchar(64) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Schema_Key_Info`
--

LOCK TABLES `Schema_Key_Info` WRITE;
/*!40000 ALTER TABLE `Schema_Key_Info` DISABLE KEYS */;
INSERT INTO `Schema_Key_Info` VALUES ('cars','car_id',NULL,NULL,'PRIMARY'),('car_in_frame','car_id',NULL,NULL,'PRIMARY'),('car_in_frame','car_id','cars','car_id','car_in_frame_ibfk_2'),('car_in_frame','f_id',NULL,NULL,'PRIMARY'),('car_in_frame','f_id','frames','f_id','car_in_frame_ibfk_1'),('ego_cars','ego_id',NULL,NULL,'PRIMARY'),('ego_frame','ego_id',NULL,NULL,'PRIMARY'),('ego_frame','ego_id','ego_cars','ego_id','ego_frame_ibfk_2'),('ego_frame','f_id',NULL,NULL,'PRIMARY'),('ego_frame','f_id','frames','f_id','ego_frame_ibfk_1'),('frames','f_id',NULL,NULL,'PRIMARY'),('succ_frame','f_id2',NULL,NULL,'PRIMARY'),('succ_frame','f_id2','frames','f_id','succ_frame_ibfk_2'),('succ_frame','f_id1',NULL,NULL,'PRIMARY'),('succ_frame','f_id1','frames','f_id','succ_frame_ibfk_1');
/*!40000 ALTER TABLE `Schema_Key_Info` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Schema_Position_Info`
--

DROP TABLE IF EXISTS `Schema_Position_Info`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Schema_Position_Info` (
  `TABLE_NAME` varchar(64) NOT NULL,
  `COLUMN_NAME` varchar(64) NOT NULL,
  `ORDINAL_POSITION` bigint(21) unsigned NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Schema_Position_Info`
--

LOCK TABLES `Schema_Position_Info` WRITE;
/*!40000 ALTER TABLE `Schema_Position_Info` DISABLE KEYS */;
INSERT INTO `Schema_Position_Info` VALUES ('cars','car_id',1),('car_in_frame','f_id',1),('car_in_frame','speed_diff',3),('car_in_frame','Lane',5),('car_in_frame','car_id',2),('car_in_frame','near_level',4),('car_in_frame','Speed',6),('ego_cars','ego_id',1),('ego_frame','f_id',1),('ego_frame','Speed',3),('ego_frame','ego_id',2),('frames','f_id',1),('succ_frame','f_id2',2),('succ_frame','f_id1',1);
/*!40000 ALTER TABLE `Schema_Position_Info` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `SelfRelationships`
--

DROP TABLE IF EXISTS `SelfRelationships`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `SelfRelationships` (
  `TABLE_NAME` varchar(64) NOT NULL,
  `REFERENCED_TABLE_NAME` varchar(64),
  `REFERENCED_COLUMN_NAME` varchar(64),
  PRIMARY KEY (`TABLE_NAME`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `SelfRelationships`
--

LOCK TABLES `SelfRelationships` WRITE;
/*!40000 ALTER TABLE `SelfRelationships` DISABLE KEYS */;
INSERT INTO `SelfRelationships` VALUES ('succ_frame','frames','f_id');
/*!40000 ALTER TABLE `SelfRelationships` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `TargetNode`
--

DROP TABLE IF EXISTS `TargetNode`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `TargetNode` (
  `Fid` varchar(199) NOT NULL,
  PRIMARY KEY (`Fid`),
  CONSTRAINT `TargetNode_ibfk_1` FOREIGN KEY (`Fid`) REFERENCES `FNodes` (`Fid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `TargetNode`
--

LOCK TABLES `TargetNode` WRITE;
/*!40000 ALTER TABLE `TargetNode` DISABLE KEYS */;
/*!40000 ALTER TABLE `TargetNode` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `TernaryRelations`
--

DROP TABLE IF EXISTS `TernaryRelations`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `TernaryRelations` (
  `TABLE_NAME` varchar(64) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `TernaryRelations`
--

LOCK TABLES `TernaryRelations` WRITE;
/*!40000 ALTER TABLE `TernaryRelations` DISABLE KEYS */;
/*!40000 ALTER TABLE `TernaryRelations` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `lattice_mapping`
--

DROP TABLE IF EXISTS `lattice_mapping`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `lattice_mapping` (
  `orig_rnid` varchar(300) NOT NULL,
  `short_rnid` varchar(20) NOT NULL,
  PRIMARY KEY (`orig_rnid`,`short_rnid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `lattice_mapping`
--

LOCK TABLES `lattice_mapping` WRITE;
/*!40000 ALTER TABLE `lattice_mapping` DISABLE KEYS */;
INSERT INTO `lattice_mapping` VALUES ('car_in_frame(frames0,cars0)','a'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0)','a,b'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0)','a,b,c'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','a,b,c,d'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','a,b,c,d,e'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','a,b,c,e'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0)','a,b,d'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','a,b,d,e'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),succ_frame(frames0,frames1)','a,b,e'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0)','a,c'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','a,c,d'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','a,c,d,e'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','a,c,e'),('car_in_frame(frames0,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','a,d,e'),('car_in_frame(frames0,cars0),succ_frame(frames0,frames1)','a,e'),('car_in_frame(frames1,cars0)','b'),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','b,c,d'),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','b,c,d,e'),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','b,c,e'),('car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0)','b,d'),('car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','b,d,e'),('car_in_frame(frames1,cars0),succ_frame(frames0,frames1)','b,e'),('ego_frame(frames0,ego_cars0)','c'),('ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','c,d'),('ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','c,d,e'),('ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','c,e'),('ego_frame(frames1,ego_cars0)','d'),('ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','d,e'),('succ_frame(frames0,frames1)','e');
/*!40000 ALTER TABLE `lattice_mapping` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `lattice_membership`
--

DROP TABLE IF EXISTS `lattice_membership`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `lattice_membership` (
  `name` varchar(300) NOT NULL,
  `member` varchar(300) NOT NULL,
  PRIMARY KEY (`name`,`member`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `lattice_membership`
--

LOCK TABLES `lattice_membership` WRITE;
/*!40000 ALTER TABLE `lattice_membership` DISABLE KEYS */;
INSERT INTO `lattice_membership` VALUES ('car_in_frame(frames0,cars0)','car_in_frame(frames0,cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0)','car_in_frame(frames0,cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0)','car_in_frame(frames1,cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0)','car_in_frame(frames0,cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0)','car_in_frame(frames1,cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames0,cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames1,cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames0,cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames1,cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0)','car_in_frame(frames0,cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames0,cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames0,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames0,cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0)'),('car_in_frame(frames0,cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames1,cars0)','car_in_frame(frames1,cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames1,cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames1,cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames1,cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0)'),('car_in_frame(frames1,cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('ego_frame(frames0,ego_cars0)','ego_frame(frames0,ego_cars0)'),('ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','ego_frame(frames0,ego_cars0)'),('ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','ego_frame(frames1,ego_cars0)'),('ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames0,ego_cars0)'),('ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames1,ego_cars0)'),('ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames0,ego_cars0)'),('ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('ego_frame(frames1,ego_cars0)','ego_frame(frames1,ego_cars0)'),('ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames1,ego_cars0)'),('ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('succ_frame(frames0,frames1)','succ_frame(frames0,frames1)');
/*!40000 ALTER TABLE `lattice_membership` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `lattice_rel`
--

DROP TABLE IF EXISTS `lattice_rel`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `lattice_rel` (
  `parent` varchar(300) NOT NULL,
  `child` varchar(300) NOT NULL,
  `removed` varchar(300) DEFAULT NULL,
  PRIMARY KEY (`parent`,`child`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `lattice_rel`
--

LOCK TABLES `lattice_rel` WRITE;
/*!40000 ALTER TABLE `lattice_rel` DISABLE KEYS */;
INSERT INTO `lattice_rel` VALUES ('car_in_frame(frames0,cars0)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0)','car_in_frame(frames1,cars0)'),('car_in_frame(frames0,cars0)','car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames0,cars0)','car_in_frame(frames0,cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0)','car_in_frame(frames1,cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0)','car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0)','car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames1,cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0)'),('car_in_frame(frames0,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames0,cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0)'),('car_in_frame(frames0,cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames0,cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames1,cars0)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0)','car_in_frame(frames0,cars0)'),('car_in_frame(frames1,cars0)','car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames1,cars0)','car_in_frame(frames1,cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames0,cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames1,ego_cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames0,cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0)'),('car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames1,cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0)'),('car_in_frame(frames1,cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames0,ego_cars0)'),('car_in_frame(frames1,cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames1,ego_cars0)'),('ego_frame(frames0,ego_cars0)','car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0)','car_in_frame(frames0,cars0)'),('ego_frame(frames0,ego_cars0)','ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','ego_frame(frames1,ego_cars0)'),('ego_frame(frames0,ego_cars0)','ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames0,cars0)'),('ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames1,cars0)'),('ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0)'),('ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0)'),('ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0)'),('ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0)'),('ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames1,ego_cars0)'),('ego_frame(frames1,ego_cars0)','car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0)','car_in_frame(frames1,cars0)'),('ego_frame(frames1,ego_cars0)','ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)','ego_frame(frames0,ego_cars0)'),('ego_frame(frames1,ego_cars0)','ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0)'),('ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0)'),('ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames0,ego_cars0)'),('EmptySet','car_in_frame(frames0,cars0)','car_in_frame(frames0,cars0)'),('EmptySet','car_in_frame(frames1,cars0)','car_in_frame(frames1,cars0)'),('EmptySet','ego_frame(frames0,ego_cars0)','ego_frame(frames0,ego_cars0)'),('EmptySet','ego_frame(frames1,ego_cars0)','ego_frame(frames1,ego_cars0)'),('EmptySet','succ_frame(frames0,frames1)','succ_frame(frames0,frames1)'),('succ_frame(frames0,frames1)','car_in_frame(frames0,cars0),succ_frame(frames0,frames1)','car_in_frame(frames0,cars0)'),('succ_frame(frames0,frames1)','car_in_frame(frames1,cars0),succ_frame(frames0,frames1)','car_in_frame(frames1,cars0)'),('succ_frame(frames0,frames1)','ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames0,ego_cars0)'),('succ_frame(frames0,frames1)','ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)','ego_frame(frames1,ego_cars0)');
/*!40000 ALTER TABLE `lattice_rel` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `lattice_set`
--

DROP TABLE IF EXISTS `lattice_set`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `lattice_set` (
  `name` varchar(300) NOT NULL,
  `length` int(11) NOT NULL,
  PRIMARY KEY (`name`,`length`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `lattice_set`
--

LOCK TABLES `lattice_set` WRITE;
/*!40000 ALTER TABLE `lattice_set` DISABLE KEYS */;
INSERT INTO `lattice_set` VALUES ('car_in_frame(frames0,cars0)',1),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0)',2),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0)',3),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)',4),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)',5),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)',4),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0)',3),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)',4),('car_in_frame(frames0,cars0),car_in_frame(frames1,cars0),succ_frame(frames0,frames1)',3),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0)',2),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)',3),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)',4),('car_in_frame(frames0,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)',3),('car_in_frame(frames0,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)',3),('car_in_frame(frames0,cars0),succ_frame(frames0,frames1)',2),('car_in_frame(frames1,cars0)',1),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)',3),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)',4),('car_in_frame(frames1,cars0),ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)',3),('car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0)',2),('car_in_frame(frames1,cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)',3),('car_in_frame(frames1,cars0),succ_frame(frames0,frames1)',2),('ego_frame(frames0,ego_cars0)',1),('ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0)',2),('ego_frame(frames0,ego_cars0),ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)',3),('ego_frame(frames0,ego_cars0),succ_frame(frames0,frames1)',2),('ego_frame(frames1,ego_cars0)',1),('ego_frame(frames1,ego_cars0),succ_frame(frames0,frames1)',2),('succ_frame(frames0,frames1)',1);
/*!40000 ALTER TABLE `lattice_set` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2024-03-11 22:49:19
